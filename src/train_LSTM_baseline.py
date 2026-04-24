"""
Baseline LSTM training pipeline for StockVolatilitySight.

Flow
----
1. Load train/val/test parquet splits from data/processed/.
2. Select input feature columns (configurable via --features).
3. Log-transform the realized-volatility target (if not already on log scale).
4. Fit a StandardScaler on train features; transform val/test.
5. Build sliding-window Datasets/DataLoaders.
6. Run an Optuna study to find the best hyperparameters, scoring each trial by
   MSE on the validation set in the *original* (inverse-log) target scale.
7. Retrain the model with the best hyperparameters on the full training set.
8. Evaluate the final model on the test set. Report MSE / RMSE / MAE.

Artifacts saved to models/:
  - lstm_baseline.pt       (state_dict + hyperparameters + scaler info)
  - lstm_baseline_scaler.joblib

Example
-------
    python train_LSTM_baseline.py --n-trials 20
    python train_LSTM_baseline.py --output-prefix lstm_core --features log_return abs_return
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

# Make config.py and src.* importable regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.lstm_model import LSTMRegressor, VolatilityWindowDataset

logger = logging.getLogger("train_LSTM_baseline")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    val = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")
    return train, val, test


def prepare_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and target from df, dropping rows with NaNs in any of
    the required columns. Returns (features, target_raw) as numpy arrays.
    """
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from dataframe: {missing}")

    sub = df[feature_cols + [target_col]].dropna()
    return sub[feature_cols].to_numpy(dtype=np.float64), sub[target_col].to_numpy(
        dtype=np.float64
    )


def maybe_log_transform(target: np.ndarray) -> np.ndarray:
    """
    Apply np.log to the realized-vol target if it is not already on a log
    scale. Realized volatility is strictly positive (~0.003 – 0.06 for SPY),
    so any non-positive value is a signal it was already log-transformed.
    """
    if np.any(target <= 0):
        logger.info("Target appears already log-transformed (non-positive values present).")
        return target
    return np.log(target)


# ── Training helpers ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def val_mse_raw(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """MSE on the validation set, computed back in the original (exp) scale."""
    pred_log, target_log = predict(model, loader, device)
    pred_raw = np.exp(pred_log)
    target_raw = np.exp(target_log)
    return float(np.mean((pred_raw - target_raw) ** 2))


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    epochs: int,
    patience: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    """
    Train with Adam + MSE on log-target. Early-stops on val MSE (raw scale).
    Returns (best_model_state_loaded, best_val_mse_raw).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        v = val_mse_raw(model, val_loader, device)
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if verbose:
            logger.info(
                "epoch %3d | train_log_mse=%.6f | val_mse_raw=%.8f | best=%.8f",
                epoch, train_loss, v, best_val,
            )
        if stale >= patience:
            if verbose:
                logger.info("Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ── Pipeline assembly ─────────────────────────────────────────────────────────

def build_loaders(
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val_log: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = VolatilityWindowDataset(X_train_scaled, y_train_log, seq_len)
    val_ds = VolatilityWindowDataset(X_val_scaled, y_val_log, seq_len)
    # Chronological order matters — do not shuffle across windows in time-series.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_optuna_study(
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val_log: np.ndarray,
    n_features: int,
    n_trials: int,
    tune_epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> optuna.Study:
    space = config.LSTM_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", space["hidden_size"])
        n_layers = trial.suggest_categorical("n_layers", space["n_layers"])
        dropout = trial.suggest_float("dropout", *space["dropout"])
        lr = trial.suggest_float("lr", *space["lr"], log=True)
        batch_size = trial.suggest_categorical("batch_size", space["batch_size"])
        seq_len = trial.suggest_categorical("seq_len", space["seq_len"])

        set_seed(seed)
        train_loader, val_loader = build_loaders(
            X_train_scaled, y_train_log,
            X_val_scaled, y_val_log,
            seq_len=seq_len, batch_size=batch_size,
        )
        model = LSTMRegressor(
            input_size=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
        _, best_val = fit_model(
            model, train_loader, val_loader,
            lr=lr, epochs=tune_epochs, patience=patience,
            device=device, verbose=True,
        )
        return best_val

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def evaluate_on_test(
    model: nn.Module,
    X_test_scaled: np.ndarray,
    y_test_log: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    test_ds = VolatilityWindowDataset(X_test_scaled, y_test_log, seq_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    pred_log, target_log = predict(model, test_loader, device)
    pred_raw = np.exp(pred_log)
    target_raw = np.exp(target_log)
    err = pred_raw - target_raw
    return {
        "MSE": float(np.mean(err ** 2)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "n_test_windows": int(len(target_raw)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline LSTM on SPY volatility.")
    p.add_argument(
        "--features", nargs="+", default=config.LSTM_BASELINE_FEATURES,
        help="Input feature column names (space-separated). "
             f"Default: {config.LSTM_BASELINE_FEATURES}",
    )
    p.add_argument(
        "--target", default=config.LSTM_TARGET,
        help=f"Target column (default: {config.LSTM_TARGET})",
    )
    p.add_argument("--n-trials", type=int, default=config.LSTM_N_TRIALS)
    p.add_argument("--tune-epochs", type=int, default=config.LSTM_TUNE_EPOCHS)
    p.add_argument("--final-epochs", type=int, default=config.LSTM_FINAL_EPOCHS)
    p.add_argument("--patience", type=int, default=config.LSTM_PATIENCE)
    p.add_argument("--seed", type=int, default=config.LSTM_RANDOM_STATE)
    p.add_argument(
        "--output-prefix", default="lstm_baseline",
        help="Filename prefix used when saving model + scaler in models/",
    )
    return p.parse_args()


def main() -> dict:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence optuna's per-trial INFO chatter; keep our own logs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = torch.device("cpu")
    set_seed(args.seed)

    logger.info("Features: %s", args.features)
    logger.info("Target  : %s", args.target)

    # 1. Load splits.
    train_df, val_df, test_df = load_splits()
    logger.info(
        "Split sizes — train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # 2. Prepare arrays.
    X_train_raw, y_train_raw = prepare_arrays(train_df, args.features, args.target)
    X_val_raw, y_val_raw = prepare_arrays(val_df, args.features, args.target)
    X_test_raw, y_test_raw = prepare_arrays(test_df, args.features, args.target)

    # 3. Log-transform target.
    y_train_log = maybe_log_transform(y_train_raw)
    y_val_log = maybe_log_transform(y_val_raw)
    y_test_log = maybe_log_transform(y_test_raw)

    # 4. Scale features (fit on train only).
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    n_features = X_train.shape[1]
    logger.info("n_features=%d | rows — train=%d val=%d test=%d",
                n_features, len(X_train), len(X_val), len(X_test))

    # 5. Optuna study.
    logger.info("Starting Optuna study with %d trials…", args.n_trials)
    study = run_optuna_study(
        X_train, y_train_log, X_val, y_val_log,
        n_features=n_features,
        n_trials=args.n_trials,
        tune_epochs=args.tune_epochs,
        patience=args.patience,
        device=device,
        seed=args.seed,
    )
    best = study.best_params
    logger.info("Best val MSE (raw scale): %.8f", study.best_value)
    logger.info("Best params: %s", best)

    # 6. Retrain with best params on train, using val for early stopping.
    set_seed(args.seed)
    train_loader, val_loader = build_loaders(
        X_train, y_train_log, X_val, y_val_log,
        seq_len=best["seq_len"], batch_size=best["batch_size"],
    )
    model = LSTMRegressor(
        input_size=n_features,
        hidden_size=best["hidden_size"],
        n_layers=best["n_layers"],
        dropout=best["dropout"],
    ).to(device)
    model, best_val_final = fit_model(
        model, train_loader, val_loader,
        lr=best["lr"], epochs=args.final_epochs, patience=args.patience,
        device=device, verbose=True,
    )
    logger.info("Final retrain — best val MSE (raw scale): %.8f", best_val_final)

    # 7. Test evaluation.
    test_metrics = evaluate_on_test(
        model, X_test, y_test_log,
        seq_len=best["seq_len"], batch_size=best["batch_size"],
        device=device,
    )
    logger.info("Test metrics: %s", test_metrics)

    # 8. Persist artifacts.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"{args.output_prefix}.pt"
    scaler_path = config.MODELS_DIR / f"{args.output_prefix}_scaler.joblib"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": best,
            "features": args.features,
            "target": args.target,
            "n_features": n_features,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    logger.info("Saved model → %s", model_path)
    logger.info("Saved scaler → %s", scaler_path)

    result = {
        "best_params": best,
        "best_val_mse_raw": float(study.best_value),
        "retrained_val_mse_raw": float(best_val_final),
        "test_metrics": test_metrics,
        "features": args.features,
        "target": args.target,
        "n_trials": args.n_trials,
    }
    print("\n=== Baseline LSTM results ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
