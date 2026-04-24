"""
HAR-RV benchmark — adapted to our 21-day-forward realized-volatility target.

Background
----------
Corsi's (2002/2004) Heterogeneous Autoregressive model of Realized Volatility
(HAR-RV) is the canonical econometric benchmark for multi-horizon realized-vol
forecasting. The 21-day-ahead variant (Andersen-Bollerslev-Diebold 2007,
Corsi-Renò 2012, Bollerslev-Patton-Quaedvlieg 2016) is the literature's
standard benchmark at exactly our target horizon.

See ``supplementary/discussion.md`` → entry dated 2026-04-24 "HAR-RV benchmark
design: target-formula alignment" for the full reasoning.

Formulation
-----------
    y_t = realized_vol_21d(t)                       -- OUR EXACT TARGET,
                                                       unchanged from features.py::
                                                       add_realized_volatility()

    y_t ≈ c + β_d · RV_d(t) + β_w · RV_w(t) + β_m · RV_m(t)

where the backward-looking predictor components at time *t* use only returns
observed up to *t*:

    RV_d(t) = |r_t|                                 -- un-demeaned, since the
                                                       1-value mean is r_t itself
                                                       (demeaning is degenerate).
                                                       Matches Corsi's daily-RV
                                                       convention.

    RV_w(t) = sqrt( (1/5)  · Σ_{j=t-4..t}  (r_j - r̄_5)² )   -- 5-day demeaned
    RV_m(t) = sqrt( (1/21) · Σ_{j=t-20..t} (r_j - r̄_21)² )  -- 21-day demeaned

The 5- and 21-day components use the same demeaned Andersen-Bollerslev formula
as the target, applied at a shorter backward window. The 1-day falls back to
``|r_t|`` because demeaning a single observation gives 0.

Fit
---
Plain OLS on train split. No hyperparameter tuning, no log transform of the
target (OLS doesn't need it; the target is already well-scaled and non-negative).

Outputs
-------
    data/processed/test_predictions_harrv.parquet  (columns: har_rv, target,
                                                    rv_d, rv_w, rv_m)

Also reports test metrics and Diebold-Mariano tests vs the naive
``rolling_std_21`` persistence baseline (same DM helper as nb 06 / gate_stacking).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.utils import regression_metrics

logger = logging.getLogger("har_rv")


HAR_FEATURE_COLS = ["rv_d", "rv_w", "rv_m"]


# ── HAR-RV predictor components ──────────────────────────────────────────────

def _demeaned_rolling_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """
    Backward-looking realized vol over *window* past days, demeaned, 1/m normed.

    Formula: sqrt( (1/m) · Σ_{j=t-m+1..t} (r_j - r̄_m)² )

    Uses the variance identity Var(X) = E[X²] - E[X]² for a fast rolling
    computation rather than rolling().apply(). Clips tiny negative values
    introduced by float noise before sqrt.
    """
    mean = log_returns.rolling(window, min_periods=window).mean()
    sq_mean = (log_returns ** 2).rolling(window, min_periods=window).mean()
    var = (sq_mean - mean ** 2).clip(lower=0)
    return np.sqrt(var)


def build_har_components(
    split_df: pd.DataFrame,
    log_return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Return *split_df* with three HAR-RV predictor columns added: rv_d, rv_w, rv_m.

    - ``rv_d``: |r_t|, un-demeaned (1-day demeaning is degenerate).
    - ``rv_w``: 5-day backward demeaned vol, matching target's Andersen-Bollerslev form.
    - ``rv_m``: 21-day backward demeaned vol, matching target's AB form.

    Input rows whose rolling window is not yet fully populated receive NaN.
    """
    if log_return_col not in split_df.columns:
        raise KeyError(
            f"Column {log_return_col!r} missing from split. Required for HAR-RV."
        )

    r = split_df[log_return_col]
    df = split_df.copy()
    df["rv_d"] = r.abs()
    df["rv_w"] = _demeaned_rolling_vol(r, window=5)
    df["rv_m"] = _demeaned_rolling_vol(r, window=21)
    return df


# ── Diebold-Mariano (same as nb 06 §6 / gate_stacking) ───────────────────────

def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    h: int = 21,
    loss: str = "mse",
) -> dict:
    """
    Diebold-Mariano test with HLN small-sample correction, Bartlett HAC lag h-1.

    Positive dm_stat ⇒ model A has higher loss ⇒ model B is more accurate.
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
        raise ValueError(f"unknown loss {loss!r}")

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


# ── Main fit/predict pipeline ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HAR-RV benchmark — OLS of realized_vol_21d on 1/5/21-day backward "
            "RV components. Fits on train split, evaluates on test, writes "
            "test_predictions_harrv.parquet and reports DM vs naive."
        )
    )
    p.add_argument(
        "--output-path", default=None,
        help=(
            "Destination parquet. Default → "
            "data/processed/test_predictions_harrv.parquet."
        ),
    )
    p.add_argument(
        "--target", default=config.LSTM_TARGET,
        help=f"Target column to predict. Default: {config.LSTM_TARGET}.",
    )
    return p.parse_args()


def main() -> dict:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_path = (
        Path(args.output_path) if args.output_path
        else config.DATA_PROCESSED / "test_predictions_harrv.parquet"
    )

    # ── Load splits ──────────────────────────────────────────────────────────
    train_df = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    test_df = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")

    for name, df in [("train", train_df), ("test", test_df)]:
        for col in ("log_return", args.target):
            if col not in df.columns:
                raise KeyError(
                    f"Column {col!r} missing from {name} split. "
                    "Re-run notebook 01 to regenerate splits."
                )

    # ── Build HAR components on both splits ──────────────────────────────────
    train_har = build_har_components(train_df)
    test_har = build_har_components(test_df)

    train_fit = train_har[HAR_FEATURE_COLS + [args.target]].dropna()
    test_eval = test_har[HAR_FEATURE_COLS + [args.target]].dropna()

    if "rolling_std_21" in test_df.columns:
        test_eval = test_eval.join(
            test_df["rolling_std_21"].rename("naive"), how="left"
        )
    else:
        # Fallback: naive == rv_m (by construction these are very close; rolling_std_21
        # uses ddof=1, rv_m uses ddof=0, tiny numerical diff). Worth warning.
        logger.warning(
            "rolling_std_21 not found on test split; using rv_m as naive for DM."
        )
        test_eval["naive"] = test_eval["rv_m"]

    logger.info(
        "Fit rows: %d, Eval rows: %d (target=%s)",
        len(train_fit), len(test_eval), args.target,
    )

    # ── Fit OLS ──────────────────────────────────────────────────────────────
    X_train = train_fit[HAR_FEATURE_COLS].values
    y_train = train_fit[args.target].values
    model = LinearRegression(fit_intercept=True).fit(X_train, y_train)

    print("\n" + "=" * 64)
    print(" HAR-RV — fitted OLS coefficients")
    print("=" * 64)
    print(f"  intercept : {model.intercept_:+.6e}")
    for feat, coef in zip(HAR_FEATURE_COLS, model.coef_):
        print(f"  {feat:<9s} : {coef:+.6e}")

    # ── Predict on test ──────────────────────────────────────────────────────
    X_test = test_eval[HAR_FEATURE_COLS].values
    y_test = test_eval[args.target].values
    har_pred = model.predict(X_test)
    naive_pred = test_eval["naive"].values

    out_df = test_eval.copy()
    out_df["har_rv"] = har_pred
    out_df = out_df.rename(columns={args.target: "target"})
    out_df = out_df[["har_rv", "target", "rv_d", "rv_w", "rv_m", "naive"]]

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n── Test-set metrics ──")
    rows = []
    for name, pred in [("naive", naive_pred), ("har_rv", har_pred)]:
        m = regression_metrics(y_test, pred)
        rows.append((name, m))
        mape_str = f"{m['MAPE']:.2f}%" if "MAPE" in m else "n/a"
        print(
            f"  {name:<8s} MSE={m['MSE']:.6e}  RMSE={m['RMSE']:.6f}  "
            f"MAE={m['MAE']:.6f}  MAPE={mape_str}"
        )

    # ── Diebold-Mariano vs naive ─────────────────────────────────────────────
    print("\n── Diebold-Mariano: HAR-RV vs Naive (rolling_std_21) ──")
    dm_results = {}
    for loss in ("mse", "mae"):
        # Sign convention: first arg is "model A", second is "model B".
        # positive dm → A (naive) has higher loss → HAR-RV wins.
        r = diebold_mariano(y_test, naive_pred, har_pred, loss=loss)
        sig = "***" if r["p_value"] < 0.01 else (
            "**"  if r["p_value"] < 0.05 else (
            "*"   if r["p_value"] < 0.10 else "n.s."
        ))
        if r["p_value"] < 0.05:
            winner = "HAR-RV wins" if r["dm_stat"] > 0 else "Naive wins"
        else:
            winner = "tie"
        print(
            f"  Naive vs HAR-RV  [{loss.upper()}]  DM={r['dm_stat']:+.2f}  "
            f"p={r['p_value']:.3f}  {sig:<4s}  {winner}"
        )
        dm_results[f"naive_vs_harrv_{loss}"] = r

    # ── Save predictions ─────────────────────────────────────────────────────
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    logger.info("Saved HAR-RV predictions → %s", out_path)

    results = {
        "intercept": float(model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(HAR_FEATURE_COLS, model.coef_)},
        "test_metrics": {
            name: {k: float(v) for k, v in m.items()} for name, m in rows
        },
        "dm_tests": {
            k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv)
                for kk, vv in r.items()}
            for k, r in dm_results.items()
        },
        "output_path": str(out_path),
        "n_train": int(len(train_fit)),
        "n_test": int(len(test_eval)),
    }
    return results


if __name__ == "__main__":
    main()
