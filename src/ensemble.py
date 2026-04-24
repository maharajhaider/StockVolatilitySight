"""
Phase 4: Ensemble Prediction

Combines the predictions of the regime-specific LSTMs (calm and volatile)
weighted by the HMM's soft probabilities for each predicted day.
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.lstm_model import LSTMRegressor, VolatilityWindowDataset
from src.train_LSTM_baseline import maybe_log_transform, predict
from src.utils import regression_metrics

logger = logging.getLogger("ensemble")


def get_aligned_predictions(
    model_name: str, test_df: pd.DataFrame, device: torch.device,
    suffix: str = "",
) -> pd.Series:
    """
    Load a trained LSTM and return its raw-scale predictions aligned to the
    correct temporal index of `test_df`.

    The *suffix* argument selects variant-specific checkpoints, e.g.
    suffix='_B' → models/lstm_{model_name}_B.pt.
    """
    pt_path = config.MODELS_DIR / f"lstm_{model_name}{suffix}.pt"
    scaler_path = config.MODELS_DIR / f"lstm_{model_name}{suffix}_scaler.joblib"

    if not pt_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing artifacts for {model_name}{suffix}.")

    logger.info("Loading model '%s'...", model_name)
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    hyper = ckpt["hyperparameters"]
    features = ckpt["features"]
    seq_len = hyper["seq_len"]

    model = LSTMRegressor(
        input_size=ckpt["n_features"],
        hidden_size=hyper["hidden_size"],
        n_layers=hyper["n_layers"],
        dropout=hyper["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    # Replicate prep logic exactly so rows match
    sub = test_df[features + [config.LSTM_TARGET]].dropna()
    X_raw = sub[features].to_numpy()
    y_raw = sub[config.LSTM_TARGET].to_numpy()

    X_scaled = scaler.transform(X_raw)
    y_log = maybe_log_transform(y_raw)

    ds = VolatilityWindowDataset(X_scaled, y_log, seq_len)
    loader = DataLoader(ds, batch_size=hyper["batch_size"], shuffle=False)

    pred_log, target_log = predict(model, loader, device)
    pred_raw = np.exp(pred_log)

    # The prediction for window ending at i is target index i
    # The dataset outputs lengths of N - seq_len + 1, matching the last elements.
    target_indices = sub.index[seq_len - 1 :]
    
    assert len(pred_raw) == len(target_indices), "Index alignment mismatch"
    
    return pd.Series(pred_raw, index=target_indices, name=model_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute ensemble test predictions from trained LSTM checkpoints."
    )
    p.add_argument(
        "--variant-suffix", default="",
        help=(
            "Suffix used when looking up LSTM checkpoints, e.g. '_B' → "
            "models/lstm_baseline_B.pt. Output is written to "
            "data/processed/test_predictions{suffix}.parquet "
            "(empty → test_predictions.parquet)."
        ),
    )
    p.add_argument(
        "--regime-probs-path", default=None,
        help=(
            "Path to regime probabilities parquet used for ensemble weighting. "
            "Default None → data/processed/regime_probabilities.parquet. "
            "For variant O: data/processed/regime_probabilities_O.parquet."
        ),
    )
    p.add_argument(
        "--hmm-meta-path", default=None,
        help=(
            "Path to HMM meta joblib. Default None → models/hmm_meta.joblib. "
            "For variant O: models/hmm_meta_O.joblib."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    suffix = args.variant_suffix

    regime_probs_path = Path(args.regime_probs_path) if args.regime_probs_path else (
        config.DATA_PROCESSED / "regime_probabilities.parquet"
    )
    hmm_meta_path = Path(args.hmm_meta_path) if args.hmm_meta_path else (
        config.MODELS_DIR / "hmm_meta.joblib"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device("cpu")

    logger.info(
        "Loading required data splits (variant_suffix='%s', regime_probs=%s, hmm_meta=%s)...",
        suffix or "<none>", regime_probs_path, hmm_meta_path,
    )
    test_df = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")
    rp = pd.read_parquet(regime_probs_path)

    # The HMM states were mapped such that we saved probabilities as `p_calm` and `p_volatile`
    meta = joblib.load(hmm_meta_path)

    logger.info("Extracting aligned predictions for all LSTMs...")
    # Get predictions for all windows in the test set.
    # Note: LSTMs with different seq_len will have their predictions start on
    # slightly different dates. Pandas outer join will naturally align them.
    pred_base = get_aligned_predictions("baseline", test_df, device, suffix=suffix)
    pred_calm = get_aligned_predictions("calm",     test_df, device, suffix=suffix)
    pred_vol  = get_aligned_predictions("volatile", test_df, device, suffix=suffix)

    # 1. Align the predictions
    df_pred = pd.concat([pred_base, pred_calm, pred_vol], axis=1)
    df_pred["target"] = test_df[config.LSTM_TARGET]
    
    # Drop rows where we lack predictions from any model (due to the max seq_len warmup)
    # or lack a target (forward Vol is NaN at end of test set)
    df_pred = df_pred.dropna()
    
    # 2. Add HMM probabilities
    # We join probabilities. The index is the prediction date `t`.
    df_eval = df_pred.join(rp[["p_calm", "p_volatile"]], how="inner")
    
    # 3. Compute Soft Ensemble
    df_eval["ensemble"] = (
        df_eval["p_calm"] * df_eval["calm"] + 
        df_eval["p_volatile"] * df_eval["volatile"]
    )
    
    # 4. Save test predictions to parquet for Phase 5 analysis
    out_path = config.DATA_PROCESSED / f"test_predictions{suffix}.parquet"
    df_eval.to_parquet(out_path)
    logger.info("Saved ensemble predictions to %s", out_path)

    # 5. Quick Terminal Report (Phase 5 will do a deeper dive)
    print("\n" + "="*50)
    print("PHASE 4: ENSEMBLE EVALUATION (TEST SET)")
    print("="*50)
    print(f"Test period: {df_eval.index.min().date()} to {df_eval.index.max().date()}")
    print(f"Test size:   {len(df_eval)} trading days")
    print("-" * 50)
    
    for model in ["baseline", "calm", "volatile", "ensemble"]:
        metrics = regression_metrics(df_eval["target"].values, df_eval[model].values)
        print(f"Model: {model.upper()}")
        print(f"  MSE:  {metrics['MSE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        if "MAPE" in metrics:
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()
