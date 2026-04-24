"""
Phase 1: Learned gating via linear stacking (meta-learner).

Replaces the HMM soft-probability weights in the ensemble with a learned
linear combination of:
    [baseline_pred, calm_pred, volatile_pred, p_calm, p_volatile]  ->  target

The fitted coefficients are directly interpretable:
  - coef on `baseline`, `calm`, `volatile` = how much weight each LSTM gets.
  - coef on `p_calm`, `p_volatile` = whether the HMM state probabilities add
    any linear information beyond the LSTM predictions themselves.

Both an unregularised OLS and a Ridge-regularised variant (alpha sweep on val)
are fit. Final test predictions are saved to
    data/processed/test_predictions_gated.parquet
and compared against the baseline LSTM and HMM-probability ensemble using
Diebold-Mariano tests (same convention as nb 06 §6).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.ensemble import get_aligned_predictions
from src.utils import regression_metrics

logger = logging.getLogger("gate_stacking")


GATE_FEATURES = ["baseline", "calm", "volatile", "p_calm", "p_volatile"]


# ── Diebold-Mariano (same as nb 06 §6) ────────────────────────────────────────

def diebold_mariano(y_true, y_pred_a, y_pred_b, h: int = 21, loss: str = "mse") -> dict:
    """Diebold-Mariano test with HLN small-sample correction, Bartlett HAC lag h-1.

    Positive dm_stat => model A has higher loss => model B wins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_a = np.asarray(y_pred_a, dtype=float)
    y_b = np.asarray(y_pred_b, dtype=float)
    if loss == "mse":
        e_a = (y_a - y_true) ** 2
        e_b = (y_b - y_true) ** 2
    elif loss == "mae":
        e_a = np.abs(y_a - y_true)
        e_b = np.abs(y_b - y_true)
    else:
        raise ValueError(f"unknown loss '{loss}'")
    d = e_a - e_b
    n = len(d)
    d_bar = float(d.mean())
    max_lag = max(h - 1, 0)
    gamma0 = float(np.var(d, ddof=0))
    S = gamma0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1)
        gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        S += 2.0 * w * gamma_k
    if S <= 0:
        S = gamma0
    dm_raw = d_bar / np.sqrt(S / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = float(dm_raw * hln)
    p_value = float(2.0 * (1.0 - sp_stats.t.cdf(np.abs(dm_stat), df=n - 1)))
    return {"dm_stat": dm_stat, "p_value": p_value, "n": n}


# ── Build prediction dataframe for any split ──────────────────────────────────

def build_eval_df(split_df: pd.DataFrame, rp: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Run all three trained LSTMs on *split_df*, join HMM soft probabilities.

    Returns a DataFrame with columns:
        baseline, calm, volatile, target, p_calm, p_volatile, hmm_ensemble
    indexed by date.
    """
    pred_base = get_aligned_predictions("baseline", split_df, device)
    pred_calm = get_aligned_predictions("calm", split_df, device)
    pred_vol = get_aligned_predictions("volatile", split_df, device)

    df = pd.concat([pred_base, pred_calm, pred_vol], axis=1)
    df.columns = ["baseline", "calm", "volatile"]
    df["target"] = split_df[config.LSTM_TARGET]
    df = df.dropna()
    df = df.join(rp[["p_calm", "p_volatile"]], how="inner")
    df["hmm_ensemble"] = df["p_calm"] * df["calm"] + df["p_volatile"] * df["volatile"]
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device("cpu")

    logger.info("Loading splits + HMM regime probabilities…")
    train_df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    val_df = pd.read_parquet(config.DATA_PROCESSED / "val.parquet")
    test_df = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")
    rp = pd.read_parquet(config.DATA_PROCESSED / "regime_probabilities.parquet")

    logger.info("Running trained LSTMs on train / val / test…")
    df_train = build_eval_df(train_df, rp, device)
    df_val = build_eval_df(val_df, rp, device)
    df_test = build_eval_df(test_df, rp, device)
    logger.info("  train rows: %d", len(df_train))
    logger.info("  val   rows: %d", len(df_val))
    logger.info("  test  rows: %d", len(df_test))

    X_train, y_train = df_train[GATE_FEATURES].values, df_train["target"].values
    X_val,   y_val   = df_val[GATE_FEATURES].values,   df_val["target"].values
    X_test,  y_test  = df_test[GATE_FEATURES].values,  df_test["target"].values

    # ── Alpha sweep on val MSE ────────────────────────────────────────────────
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    sweep = []
    best = {"val_mse": float("inf"), "alpha": None, "model": None}

    for alpha in alphas:
        if alpha == 0.0:
            m = LinearRegression(fit_intercept=True).fit(X_train, y_train)
        else:
            m = Ridge(alpha=alpha, fit_intercept=True).fit(X_train, y_train)
        val_mse = float(np.mean((m.predict(X_val) - y_val) ** 2))
        sweep.append({"alpha": alpha, "val_mse": val_mse})
        if val_mse < best["val_mse"]:
            best = {"val_mse": val_mse, "alpha": alpha, "model": m}

    print("\n" + "=" * 64)
    print(" PHASE 1 — LINEAR STACKING GATE")
    print("=" * 64)

    print("\n── Alpha sweep (val MSE) ──")
    for row in sweep:
        marker = "  ← best" if row["alpha"] == best["alpha"] else ""
        label = "OLS" if row["alpha"] == 0.0 else f"Ridge α={row['alpha']:g}"
        print(f"  {label:<14s}  val_MSE = {row['val_mse']:.6e}{marker}")

    # ── Learned coefficients ─────────────────────────────────────────────────
    best_model = best["model"]
    print("\n── Learned gate (best model) ──")
    print(f"  intercept     : {best_model.intercept_:+.6e}")
    for feat, coef in zip(GATE_FEATURES, best_model.coef_):
        print(f"  {feat:<13s} : {coef:+.6e}")

    # ── Also report unregularised OLS coefficients for interpretability ──────
    ols = LinearRegression(fit_intercept=True).fit(X_train, y_train)
    print("\n── OLS coefficients (unregularised, for comparison) ──")
    print(f"  intercept     : {ols.intercept_:+.6e}")
    for feat, coef in zip(GATE_FEATURES, ols.coef_):
        print(f"  {feat:<13s} : {coef:+.6e}")

    # ── Test-set metrics ─────────────────────────────────────────────────────
    gated_pred = best_model.predict(X_test)
    df_test["gated"] = gated_pred

    naive_pred = test_df["rolling_std_21"].reindex(df_test.index).values

    print("\n── Test-set performance ──")
    rows = []
    for name, pred in [
        ("naive",        naive_pred),
        ("baseline",     df_test["baseline"].values),
        ("hmm_ensemble", df_test["hmm_ensemble"].values),
        ("gated",        gated_pred),
    ]:
        m = regression_metrics(y_test, pred)
        rows.append((name, m))
        print(f"  {name:<14s} MSE={m['MSE']:.6e}  RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  MAPE={m['MAPE']:.2f}%")

    # ── Diebold-Mariano tests on test set ────────────────────────────────────
    print("\n── Diebold-Mariano: Gated vs baselines ──")
    dm_results = {}
    comparisons = [
        ("Gated vs Naive",         gated_pred, naive_pred),
        ("Gated vs Baseline",      gated_pred, df_test["baseline"].values),
        ("Gated vs HMM-Ensemble",  gated_pred, df_test["hmm_ensemble"].values),
    ]
    for label, a, b in comparisons:
        for loss in ["mse", "mae"]:
            r = diebold_mariano(y_test, a, b, loss=loss)
            sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.10 else "n.s."
            if r["p_value"] < 0.05:
                winner = "Gated wins" if r["dm_stat"] < 0 else "Other wins"
            else:
                winner = "tie"
            print(f"  {label:<25s} [{loss.upper()}]  DM={r['dm_stat']:+.2f}  p={r['p_value']:.3f}  {sig:<4s}  {winner}")
            dm_results[f"{label}_{loss}"] = r

    # ── Save predictions for downstream use ──────────────────────────────────
    out_path = config.DATA_PROCESSED / "test_predictions_gated.parquet"
    df_test.to_parquet(out_path)
    logger.info("Saved gated test predictions → %s", out_path)

    # ── Also print a JSON results blob for programmatic parsing ──────────────
    results = {
        "best_alpha": best["alpha"],
        "val_mse_best": best["val_mse"],
        "alpha_sweep": sweep,
        "coefficients": {f: float(c) for f, c in zip(GATE_FEATURES, best_model.coef_)},
        "intercept": float(best_model.intercept_),
        "ols_coefficients": {f: float(c) for f, c in zip(GATE_FEATURES, ols.coef_)},
        "ols_intercept": float(ols.intercept_),
        "test_metrics": {name: {k: float(v) for k, v in m.items()} for name, m in rows},
        "dm_tests": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv for kk, vv in r.items()} for k, r in dm_results.items()},
    }
    print("\n=== JSON results ===")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
