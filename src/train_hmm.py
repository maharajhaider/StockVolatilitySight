"""
Programmatic HMM training — CLI counterpart to ``notebooks/03_hmm_regime.ipynb``.

Fits the GaussianHMM / GMMHMM candidate family on the train split, selects the
winner by validation BIC, saves winner + scaler + meta, and writes
soft probabilities and Viterbi labels for the full timeline to a parquet.

CLI parameterises which feature group the HMM trains on, and which output
suffix the artifacts use. Three variant-indexed use-cases (plus backward-
compat aliases):

1. Variant A (default — matches notebook 03 pre-refactor):
     python -m src.train_hmm --features-group HMM_VARIANT_A_FEATURES
   → models/hmm_winner.joblib, data/processed/regime_probabilities.parquet

2. Variant O — price/volume features only, no sentiment:
     python -m src.train_hmm --features-group HMM_VARIANT_O_FEATURES \\
                             --output-suffix _O
   → models/hmm_winner_O.joblib, data/processed/regime_probabilities_O.parquet

3. Variant B — A's features + VIX family:
     python -m src.train_hmm --features-group HMM_VARIANT_B_FEATURES \\
                             --output-suffix _B
   → models/hmm_winner_B.joblib, data/processed/regime_probabilities_B.parquet

Backward-compat aliases (same underlying lists):
    HMM_FEATURES             = HMM_VARIANT_A_FEATURES
    HMM_PRICE_VOLUME_FEATURES = HMM_VARIANT_O_FEATURES

All heavy lifting is delegated to ``src.hmm_model`` (same functions notebook 03
calls). This module is a thin CLI wrapper so variant O can be retrained head-
lessly (Colab / background job) without touching the notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.hmm_model import (
    compare_models,
    fit_scaler,
    forward_backward_proba,
    identify_volatile_state,
    save_model,
    viterbi_decode,
)

logger = logging.getLogger("train_hmm")


# Feature groups exposed on the CLI. Keys are the valid values of
# --features-group; values are the attribute names on `config`.
# Variant-indexed names are preferred; aliases preserved for backward compat.
FEATURE_GROUPS = {
    "HMM_VARIANT_O_FEATURES":    "HMM_VARIANT_O_FEATURES",
    "HMM_VARIANT_A_FEATURES":    "HMM_VARIANT_A_FEATURES",
    "HMM_VARIANT_B_FEATURES":    "HMM_VARIANT_B_FEATURES",
    # Backward-compat aliases
    "HMM_FEATURES":              "HMM_FEATURES",                # = HMM_VARIANT_A_FEATURES
    "HMM_PRICE_VOLUME_FEATURES": "HMM_PRICE_VOLUME_FEATURES",   # = HMM_VARIANT_O_FEATURES
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the HMM candidate family (Gaussian / GMMHMM) on a parameterised "
            "feature group, select winner by val BIC, save artifacts with an optional "
            "output suffix."
        )
    )
    p.add_argument(
        "--features-group",
        default="HMM_VARIANT_A_FEATURES",
        choices=sorted(FEATURE_GROUPS.keys()),
        help=(
            "Name of the config attribute that lists HMM input features. "
            "HMM_VARIANT_A_FEATURES (default) = price/vol + rolling + sentiment (13 feats). "
            "HMM_VARIANT_O_FEATURES = price/vol + rolling only (11 feats, variant O). "
            "HMM_VARIANT_B_FEATURES = A + VIX family (16 feats, variant B). "
            "Aliases: HMM_FEATURES (= A), HMM_PRICE_VOLUME_FEATURES (= O)."
        ),
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffix appended to saved artifacts, e.g. '_O'. Default '' preserves "
            "the notebook-03 filenames (hmm_winner.joblib, regime_probabilities.parquet)."
        ),
    )
    p.add_argument(
        "--n-restarts",
        type=int,
        default=config.HMM_N_RESTARTS,
        help="Number of EM restarts per candidate model (default from config).",
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=config.HMM_N_ITER,
        help="Maximum EM iterations per fit (default from config).",
    )
    p.add_argument(
        "--duration-penalty",
        type=float,
        default=config.HMM_DURATION_PENALTY,
        help="λ in penalised restart selection (default from config).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=config.HMM_RANDOM_STATE,
        help="Random seed for EM and restarts (default from config).",
    )
    return p.parse_args()


def _prepare_split(
    split_df: pd.DataFrame, feature_cols: list[str], scaler=None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop rolling-warmup NaNs, optionally scale, return (sub_df, array)."""
    sub = split_df[feature_cols].dropna()
    raw = sub.values.astype(np.float64)
    if scaler is not None:
        raw = scaler.transform(raw)
    return sub, raw


def main() -> dict:
    args = parse_args()
    suffix = args.output_suffix

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    feature_cols: list[str] = getattr(config, FEATURE_GROUPS[args.features_group])
    logger.info(
        "HMM training — features_group=%s (%d feats), output_suffix=%s",
        args.features_group,
        len(feature_cols),
        suffix or "<none>",
    )
    logger.info("Feature list: %s", feature_cols)

    # ── Load splits ──────────────────────────────────────────────────────────
    train_df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")
    test_df = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from train.parquet: {missing}. "
            f"Re-run notebook 01 to regenerate splits with the current feature set."
        )

    # ── Fit scaler on train; transform all splits ────────────────────────────
    train_hmm_df, train_raw = _prepare_split(train_df, feature_cols)
    scaler = fit_scaler(train_raw)
    train_X = scaler.transform(train_raw)
    val_hmm_df, val_X = _prepare_split(val_df, feature_cols, scaler=scaler)
    test_hmm_df, test_X = _prepare_split(test_df, feature_cols, scaler=scaler)

    logger.info(
        "Rows after NaN drop — train: %d, val: %d, test: %d",
        len(train_X),
        len(val_X),
        len(test_X),
    )

    # ── Train and compare model family ───────────────────────────────────────
    summary_df, comparisons = compare_models(
        train_X=train_X,
        val_X=val_X,
        n_states=config.HMM_N_STATES,
        n_mix_values=config.HMM_N_MIX,
        covariance_type=config.HMM_COVARIANCE_TYPE,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_restarts=args.n_restarts,
        duration_penalty=args.duration_penalty,
    )

    print("\nModel comparison (sorted by val BIC, ascending = better):")
    print(summary_df.to_string(index=False))

    # ── Identify winner ──────────────────────────────────────────────────────
    best_label = summary_df.iloc[0]["model"]
    best_comp = next(c for c in comparisons if c.label == best_label)
    best_model = best_comp.model

    logger.info(
        "Winner: %s  (val LL=%.2f, val BIC=%.2f, k=%d)",
        best_label,
        best_comp.val_log_likelihood,
        best_comp.bic_val,
        best_comp.n_params,
    )

    # ── Relabel states (volatile vs calm) using abs_return mean ─────────────
    if "abs_return" not in feature_cols:
        raise ValueError(
            "identify_volatile_state relies on 'abs_return' being in the HMM features. "
            f"Current features: {feature_cols}"
        )
    abs_return_idx = feature_cols.index("abs_return")
    train_states = viterbi_decode(best_model, train_X)
    state_map = identify_volatile_state(
        best_model, train_X, train_states, feature_idx=abs_return_idx
    )
    volatile_state = state_map["volatile"]
    calm_state = state_map["calm"]
    logger.info("State map: %s", state_map)

    n_vol = int((train_states == volatile_state).sum())
    n_calm = int((train_states == calm_state).sum())
    logger.info(
        "Training regime mix — calm: %d (%.1f%%), volatile: %d (%.1f%%)",
        n_calm,
        100 * n_calm / len(train_states),
        n_vol,
        100 * n_vol / len(train_states),
    )

    # ── Save model + scaler + meta ───────────────────────────────────────────
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"hmm_winner{suffix}.joblib"
    scaler_path = config.MODELS_DIR / f"hmm_scaler{suffix}.joblib"
    meta_path = config.MODELS_DIR / f"hmm_meta{suffix}.joblib"

    save_model(best_model, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    meta = {
        "winning_label": best_label,
        "n_states": config.HMM_N_STATES,
        "hmm_features": feature_cols,
        "features_group": args.features_group,
        "volatile_state": volatile_state,
        "calm_state": calm_state,
        "val_log_likelihood": float(best_comp.val_log_likelihood),
        "val_bic": float(best_comp.bic_val),
        "val_aic": float(best_comp.aic_val),
        "n_params": int(best_comp.n_params),
        "output_suffix": suffix,
    }
    joblib.dump(meta, str(meta_path))
    logger.info("Saved: %s, %s, %s", model_path, scaler_path, meta_path)

    # ── Full-timeline soft probs + Viterbi labels ────────────────────────────
    all_X = np.vstack([train_X, val_X, test_X])
    all_dates = train_hmm_df.index.append(val_hmm_df.index).append(test_hmm_df.index)

    all_states = viterbi_decode(best_model, all_X)
    all_proba = forward_backward_proba(best_model, all_X)
    p_volatile = all_proba[:, volatile_state]
    p_calm = all_proba[:, calm_state]

    regime_proba_df = pd.DataFrame(
        {
            "p_calm": p_calm,
            "p_volatile": p_volatile,
            "viterbi_state": all_states,
            "regime_label": [
                "volatile" if s == volatile_state else "calm" for s in all_states
            ],
        },
        index=all_dates,
    )
    regime_proba_df.index.name = "Date"

    regime_proba_path = config.DATA_PROCESSED / f"regime_probabilities{suffix}.parquet"
    regime_proba_df.to_parquet(regime_proba_path)
    logger.info("Saved regime probabilities → %s (%d rows)", regime_proba_path, len(regime_proba_df))

    # ── Quick full-timeline summary ──────────────────────────────────────────
    n_vol_full = int((all_states == volatile_state).sum())
    n_calm_full = len(all_states) - n_vol_full
    pct_vol = 100 * n_vol_full / len(all_states)
    pct_calm = 100 - pct_vol
    print(
        f"\nFull-timeline regime mix — "
        f"calm: {n_calm_full:,} ({pct_calm:.1f}%), "
        f"volatile: {n_vol_full:,} ({pct_vol:.1f}%)"
    )

    return {
        "winning_label": best_label,
        "val_bic": float(best_comp.bic_val),
        "volatile_state": volatile_state,
        "calm_state": calm_state,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "meta_path": str(meta_path),
        "regime_proba_path": str(regime_proba_path),
    }


if __name__ == "__main__":
    main()
