"""
Regime-specific LSTM training pipeline for StockVolatilitySight — Phase 3b.

Each LSTM is trained on windows whose *dominant regime* (by majority Viterbi
state within the window) matches the target regime (calm or volatile).

Flow
----
1. Load train/val/test parquet splits + regime_probabilities.parquet.
2. Merge Viterbi regime labels onto each split by date index.
3. For each regime in {calm, volatile}:
   a. Build a VolatilityWindowDataset that keeps only windows whose dominant
      regime matches. (A window of length seq_len is labelled by the mode of
      the Viterbi states across all seq_len timesteps.)
   b. Run an Optuna study on the validation set (same search space as baseline).
   c. Retrain with best params using early stopping on val.
   d. Evaluate on the full test set (not filtered — so the regime LSTMs are
      scored on all test windows; per-regime breakdown is done in Phase 5).
   e. Save  models/lstm_{regime}.pt  and  models/lstm_{regime}_scaler.joblib.

Design notes
------------
- The validation filter mirrors the training filter: only windows whose
  dominant val regime matches are used for early-stopping. This prevents
  the calm LSTM from over-fitting to the volatile validation distribution.
- The test evaluation is intentionally NOT filtered, because the ensemble
  (Phase 4) will call both regime LSTMs on every test window and blend by
  HMM soft probability — so each model needs to produce *sensible* (not
  NaN or degenerate) outputs for all windows.
- Scaler is always fit on the full training split (not regime-filtered) to
  avoid shifting the scale for the model that sees only calm or volatile data.

Example
-------
    python src/train_LSTM_regime.py --n-trials 20
    python src/train_LSTM_regime.py --regime calm --n-trials 30
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.lstm_model import LSTMRegressor, VolatilityWindowDataset
from src.train_LSTM_baseline import (
    build_loaders,
    evaluate_on_test,
    fit_model,
    load_splits,
    maybe_log_transform,
    prepare_arrays,
    set_seed,
)

logger = logging.getLogger("train_LSTM_regime")

REGIMES = ("calm", "volatile")

# Minimum number of regime-filtered val windows required to use the filtered
# val loader. If fewer windows exist, fall back to the unfiltered val loader.
# This handles the volatile regime during 2016-2019 (a calm-dominated period).
MIN_VAL_WINDOWS = 1


# ── Regime-filtered dataset ───────────────────────────────────────────────────

def dominant_regime(viterbi_window: np.ndarray) -> int:
    """
    Return the most common Viterbi state within a single window.

    Parameters
    ----------
    viterbi_window : int array of shape (seq_len,) containing 0/1 state labels.

    Returns
    -------
    Dominant state (int).
    """
    counts = np.bincount(viterbi_window.astype(int), minlength=2)
    return int(np.argmax(counts))


class RegimeWindowDataset(Dataset):
    """
    Sliding-window dataset that retains only windows whose dominant Viterbi
    regime matches *target_state*.

    Parameters
    ----------
    features     : (T, F) feature array (already scaled).
    targets      : (T,) target array (log-transformed).
    viterbi      : (T,) integer array of Viterbi regime labels (0=calm, 1=volatile).
    seq_len      : window length.
    target_state : the regime state to keep (0 for calm, 1 for volatile).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        viterbi: np.ndarray,
        seq_len: int,
        target_state: int,
    ) -> None:
        if not (len(features) == len(targets) == len(viterbi)):
            raise ValueError("features, targets, and viterbi must have the same length.")
        if len(features) < seq_len:
            raise ValueError(f"Not enough rows ({len(features)}) for seq_len={seq_len}")

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.viterbi = viterbi.astype(int)
        self.seq_len = seq_len
        self.target_state = target_state

        # Pre-compute valid window indices.
        n = len(features)
        self.valid_indices: list[int] = []
        for i in range(n - seq_len + 1):
            window_states = self.viterbi[i : i + seq_len]
            if dominant_regime(window_states) == target_state:
                self.valid_indices.append(i)

        logger.info(
            "RegimeWindowDataset(regime=%s): %d / %d windows kept",
            "calm" if target_state == 0 else "volatile",
            len(self.valid_indices),
            n - seq_len + 1,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.valid_indices[idx]
        x = self.features[i : i + self.seq_len]
        y = self.targets[i + self.seq_len - 1]
        return x, y


# ── Regime label helpers ──────────────────────────────────────────────────────

def load_viterbi(
    split_df: pd.DataFrame,
    regime_probs_path: Path | None = None,
    hmm_meta_path: Path | None = None,
) -> np.ndarray:
    """
    Load Viterbi regime labels for *split_df* by joining on date index.

    Returns integer array (0=calm, 1=volatile) aligned to split_df rows
    (NaN rows filled with calm=0 as fallback).

    Optional *regime_probs_path* / *hmm_meta_path* point to variant-specific
    artifacts (e.g. variant O's ``regime_probabilities_O.parquet`` /
    ``hmm_meta_O.joblib``). Defaults match variants A / B / H.
    """
    rp_path = regime_probs_path or (config.DATA_PROCESSED / "regime_probabilities.parquet")
    meta_path = hmm_meta_path or (config.MODELS_DIR / "hmm_meta.joblib")

    rp = pd.read_parquet(rp_path)
    # Map text label → int using the HMM meta so we are consistent.
    meta = joblib.load(meta_path)
    calm_state: int = meta["calm_state"]
    volatile_state: int = meta["volatile_state"]

    # Use the integer viterbi_state column directly.
    joined = split_df[[config.LSTM_TARGET]].join(
        rp[["viterbi_state"]], how="left"
    )
    labels = joined["viterbi_state"].fillna(calm_state).astype(int).values
    return labels


# ── Regime-filtered loaders ───────────────────────────────────────────────────

def build_regime_loaders(
    X_scaled: np.ndarray,
    y_log: np.ndarray,
    viterbi: np.ndarray,
    seq_len: int,
    batch_size: int,
    target_state: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders filtered by regime.
    The val loader preserves temporal order (shuffle=False).
    """
    # Split the aligned arrays at the boundary between train and val.
    # We pass the full train & val arrays together, so we need to split here.
    # Actually we receive them separately — see caller.
    raise NotImplementedError("Call build_single_regime_loader instead.")


def build_single_regime_loader(
    X_scaled: np.ndarray,
    y_log: np.ndarray,
    viterbi: np.ndarray,
    seq_len: int,
    batch_size: int,
    target_state: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = RegimeWindowDataset(X_scaled, y_log, viterbi, seq_len, target_state)
    if len(ds) == 0:
        raise ValueError(
            f"No windows found for regime {target_state} with seq_len={seq_len}. "
            "Try a longer data range or smaller seq_len."
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_fallback_val_loader(
    X_val_scaled: np.ndarray,
    y_val_log: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    """Full (unfiltered) val loader used when regime has too few val windows."""
    from src.lstm_model import VolatilityWindowDataset
    ds = VolatilityWindowDataset(X_val_scaled, y_val_log, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ── Optuna study ──────────────────────────────────────────────────────────────

def run_regime_optuna(
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    viterbi_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val_log: np.ndarray,
    viterbi_val: np.ndarray,
    n_features: int,
    target_state: int,
    n_trials: int,
    tune_epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> optuna.Study:
    space = config.LSTM_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", space["hidden_size"])
        n_layers    = trial.suggest_categorical("n_layers",    space["n_layers"])
        dropout     = trial.suggest_float("dropout", *space["dropout"])
        lr          = trial.suggest_float("lr",      *space["lr"], log=True)
        batch_size  = trial.suggest_categorical("batch_size", space["batch_size"])
        seq_len     = trial.suggest_categorical("seq_len",    space["seq_len"])

        set_seed(seed)
        try:
            train_loader = build_single_regime_loader(
                X_train_scaled, y_train_log, viterbi_train,
                seq_len, batch_size, target_state, shuffle=True,
            )
        except ValueError as exc:
            logger.warning("Trial pruned — %s", exc)
            raise optuna.exceptions.TrialPruned() from exc

        # Prefer regime-filtered val loader; fall back to full val if too sparse.
        try:
            val_ds_tmp = RegimeWindowDataset(
                X_val_scaled, y_val_log, viterbi_val, seq_len, target_state
            )
            if len(val_ds_tmp) >= MIN_VAL_WINDOWS:
                val_loader = DataLoader(val_ds_tmp, batch_size=batch_size, shuffle=False)
            else:
                logger.warning(
                    "Val has only %d window(s) for regime %s (seq_len=%d) — "
                    "falling back to unfiltered val loader.",
                    len(val_ds_tmp), target_state, seq_len,
                )
                val_loader = build_fallback_val_loader(
                    X_val_scaled, y_val_log, seq_len, batch_size
                )
        except ValueError:
            val_loader = build_fallback_val_loader(
                X_val_scaled, y_val_log, seq_len, batch_size
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
            device=device, verbose=False,
        )
        return best_val

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# ── Per-regime pipeline ───────────────────────────────────────────────────────

def train_one_regime(
    regime: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """End-to-end training for a single regime (calm or volatile)."""
    meta = joblib.load(args.hmm_meta_path)
    target_state: int = meta[f"{regime}_state"]
    logger.info("=== Training regime LSTM: %s (state=%d) ===", regime, target_state)

    # ── 1. Prepare arrays ────────────────────────────────────────────────────
    X_train_raw, y_train_raw = prepare_arrays(train_df, args.features, args.target)
    X_val_raw,   y_val_raw   = prepare_arrays(val_df,   args.features, args.target)
    X_test_raw,  y_test_raw  = prepare_arrays(test_df,  args.features, args.target)

    # ── 2. Log-transform target ──────────────────────────────────────────────
    y_train_log = maybe_log_transform(y_train_raw)
    y_val_log   = maybe_log_transform(y_val_raw)
    y_test_log  = maybe_log_transform(y_test_raw)

    # ── 3. Scale features (fit on full train) ────────────────────────────────
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)
    n_features = X_train.shape[1]

    # ── 4. Viterbi labels aligned to each split ──────────────────────────────
    # Trim arrays to match NaN-dropped prepare_arrays output length
    # prepare_arrays drops NaN rows; we must align viterbi to the same rows.
    train_nonan_idx = train_df[args.features + [args.target]].dropna().index
    val_nonan_idx   = val_df[args.features + [args.target]].dropna().index

    rp = pd.read_parquet(args.regime_probs_path)
    viterbi_train = rp["viterbi_state"].reindex(train_nonan_idx).fillna(meta["calm_state"]).astype(int).values
    viterbi_val   = rp["viterbi_state"].reindex(val_nonan_idx).fillna(meta["calm_state"]).astype(int).values

    logger.info(
        "Regime %s: train=%d rows (%d in regime), val=%d rows (%d in regime)",
        regime, len(X_train), (viterbi_train == target_state).sum(),
        len(X_val), (viterbi_val == target_state).sum(),
    )

    # ── 5. Optuna study ──────────────────────────────────────────────────────
    logger.info("Starting Optuna study for '%s' regime (%d trials)…", regime, args.n_trials)
    study = run_regime_optuna(
        X_train, y_train_log, viterbi_train,
        X_val,   y_val_log,   viterbi_val,
        n_features=n_features,
        target_state=target_state,
        n_trials=args.n_trials,
        tune_epochs=args.tune_epochs,
        patience=args.patience,
        device=device,
        seed=args.seed,
    )
    best = study.best_params
    logger.info(
        "[%s] Best val MSE (raw scale): %.8f | params: %s",
        regime, study.best_value, best,
    )

    # ── 6. Final retrain with best params ────────────────────────────────────
    set_seed(args.seed)
    train_loader = build_single_regime_loader(
        X_train, y_train_log, viterbi_train,
        best["seq_len"], best["batch_size"], target_state, shuffle=True,
    )
    # Same fallback logic as in Optuna objective.
    try:
        val_ds_final = RegimeWindowDataset(
            X_val, y_val_log, viterbi_val, best["seq_len"], target_state
        )
        if len(val_ds_final) >= MIN_VAL_WINDOWS:
            val_loader = DataLoader(val_ds_final, batch_size=best["batch_size"], shuffle=False)
            logger.info(
                "[%s] Final retrain — using regime-filtered val loader (%d windows).",
                regime, len(val_ds_final),
            )
        else:
            val_loader = build_fallback_val_loader(
                X_val, y_val_log, best["seq_len"], best["batch_size"]
            )
            logger.info(
                "[%s] Final retrain — too few regime val windows (%d); using full val.",
                regime, len(val_ds_final),
            )
    except ValueError:
        val_loader = build_fallback_val_loader(
            X_val, y_val_log, best["seq_len"], best["batch_size"]
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
    logger.info("[%s] Final retrain — best val MSE (raw): %.8f", regime, best_val_final)

    # ── 7. Test evaluation (full test set — needed for ensemble) ─────────────
    test_metrics = evaluate_on_test(
        model, X_test, y_test_log,
        seq_len=best["seq_len"], batch_size=best["batch_size"],
        device=device,
    )
    logger.info("[%s] Test metrics: %s", regime, test_metrics)

    # ── 8. Persist ───────────────────────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"lstm_{regime}{args.output_suffix}"
    model_path  = config.MODELS_DIR / f"{prefix}.pt"
    scaler_path = config.MODELS_DIR / f"{prefix}_scaler.joblib"

    torch.save(
        {
            "state_dict":      model.state_dict(),
            "hyperparameters": best,
            "features":        args.features,
            "target":          args.target,
            "n_features":      n_features,
            "regime":          regime,
            "target_state":    target_state,
        },
        model_path,
    )
    joblib.dump(scaler, scaler_path)
    logger.info("[%s] Saved model → %s", regime, model_path)
    logger.info("[%s] Saved scaler → %s", regime, scaler_path)

    return {
        "regime":              regime,
        "best_params":         best,
        "best_val_mse_raw":    float(study.best_value),
        "retrained_val_mse":   float(best_val_final),
        "test_metrics":        test_metrics,
        "features":            args.features,
        "target":              args.target,
        "n_trials":            args.n_trials,
        "n_train_windows":     int((viterbi_train == target_state).sum()),
        "n_val_windows":       int((viterbi_val   == target_state).sum()),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train regime-specific LSTMs on HMM-filtered windows (Phase 3b)."
    )
    p.add_argument(
        "--regime", choices=list(REGIMES) + ["both"], default="both",
        help="Which regime to train ('calm', 'volatile', or 'both'). Default: both.",
    )
    p.add_argument(
        "--features", nargs="+", default=config.LSTM_BASELINE_FEATURES,
        help=f"Input feature columns. Default: {config.LSTM_BASELINE_FEATURES}",
    )
    p.add_argument("--target",       default=config.LSTM_TARGET)
    p.add_argument("--n-trials",     type=int, default=config.LSTM_N_TRIALS)
    p.add_argument("--tune-epochs",  type=int, default=config.LSTM_TUNE_EPOCHS)
    p.add_argument("--final-epochs", type=int, default=config.LSTM_FINAL_EPOCHS)
    p.add_argument("--patience",     type=int, default=config.LSTM_PATIENCE)
    p.add_argument("--seed",         type=int, default=config.LSTM_RANDOM_STATE)
    p.add_argument(
        "--output-suffix", default="",
        help=(
            "Optional suffix appended to saved-model filenames, e.g. '_B' → "
            "models/lstm_calm_B.pt. Default empty preserves the original naming."
        ),
    )
    p.add_argument(
        "--regime-probs-path", default=None,
        help=(
            "Path to regime probabilities parquet (for variant-specific HMMs). "
            "Default None → data/processed/regime_probabilities.parquet. "
            "For variant O: data/processed/regime_probabilities_O.parquet."
        ),
    )
    p.add_argument(
        "--hmm-meta-path", default=None,
        help=(
            "Path to HMM meta joblib (state→calm/volatile mapping). "
            "Default None → models/hmm_meta.joblib. "
            "For variant O: models/hmm_meta_O.joblib."
        ),
    )
    return p.parse_args()


def main() -> dict:
    args = parse_args()

    # Resolve variant-probe-path defaults now so downstream code can just read
    # args.regime_probs_path / args.hmm_meta_path without knowing about defaults.
    args.regime_probs_path = (
        Path(args.regime_probs_path) if args.regime_probs_path
        else config.DATA_PROCESSED / "regime_probabilities.parquet"
    )
    args.hmm_meta_path = (
        Path(args.hmm_meta_path) if args.hmm_meta_path
        else config.MODELS_DIR / "hmm_meta.joblib"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Regime-LSTM training — output_suffix=%s, regime_probs=%s, hmm_meta=%s",
        args.output_suffix or "<none>", args.regime_probs_path, args.hmm_meta_path,
    )

    device = torch.device("cpu")
    set_seed(args.seed)

    train_df, val_df, test_df = load_splits()
    logger.info(
        "Split sizes — train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    regimes_to_run = list(REGIMES) if args.regime == "both" else [args.regime]

    all_results = {}
    for regime in regimes_to_run:
        result = train_one_regime(
            regime, train_df, val_df, test_df, args, device
        )
        all_results[regime] = result

    print("\n=== Regime-Specific LSTM Results ===")
    print(json.dumps(all_results, indent=2))
    return all_results


if __name__ == "__main__":
    main()
