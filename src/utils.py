"""
Shared utilities: plotting helpers, regression metrics, normality tests,
and miscellaneous conveniences used across notebooks and modules.
"""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 120


def _apply_style() -> None:
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")


# ── Regression metrics ─────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute MSE, RMSE, MAE, and MAPE between *y_true* and *y_pred*.
    MAPE is only computed if no true values are exactly zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]

    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))

    results = {"MSE": mse, "RMSE": rmse, "MAE": mae}

    if np.all(yt != 0):
        results["MAPE"] = np.mean(np.abs((yt - yp) / yt)) * 100

    return results


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    """Pretty-print a metrics dict."""
    header = f"── {label} ──" if label else "── Metrics ──"
    print(header)
    for k, v in metrics.items():
        print(f"  {k:>8}: {v:.6f}")


# ── Normality tests ────────────────────────────────────────────────────────────

def normality_report(
    series: pd.Series | np.ndarray,
    name: str = "series",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Shapiro-Wilk, Jarque-Bera, and D'Agostino-Pearson normality tests
    on *series* and return a summary DataFrame.

    For large n, Shapiro-Wilk is run on a random subsample (n=5000) as it
    becomes unreliable / very slow for large samples — the p-value is nearly
    always 0 for real financial data anyway.

    Parameters
    ----------
    series : 1-D array-like of floats
    name   : label for display
    alpha  : significance level for reject/fail-to-reject decision

    Returns
    -------
    DataFrame with columns: test, statistic, p_value, reject_normality
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]

    rows = []

    # Shapiro-Wilk (subsample if needed)
    sw_arr = arr if len(arr) <= 5000 else np.random.default_rng(42).choice(arr, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sw_arr)
    rows.append(("Shapiro-Wilk", sw_stat, sw_p))

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(arr)
    rows.append(("Jarque-Bera", jb_stat, jb_p))

    # D'Agostino-Pearson (combines skewness + kurtosis)
    da_stat, da_p = stats.normaltest(arr)
    rows.append(("D'Agostino-Pearson", da_stat, da_p))

    df = pd.DataFrame(rows, columns=["test", "statistic", "p_value"])
    df["reject_normality"] = df["p_value"] < alpha

    skew = stats.skew(arr)
    kurt = stats.kurtosis(arr, fisher=True)  # excess kurtosis; 0 for normal
    print(f"\n── Normality report: {name} (n={len(arr)}) ──")
    print(f"  Skewness          : {skew:+.4f}  (0 = symmetric)")
    print(f"  Excess kurtosis   : {kurt:+.4f}  (0 = mesokurtic / normal)")
    print(df.to_string(index=False))
    print()

    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_price_and_returns(
    df: pd.DataFrame,
    close_col: str = "Close",
    return_col: str = "log_return",
    title: str = "SPY Price and Log Returns",
) -> plt.Figure:
    """Two-panel plot: price series (top) and log return series (bottom)."""
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df.index, df[close_col], linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title(title)

    axes[1].plot(df.index, df[return_col], linewidth=0.5, color="dimgray", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Log Return")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    return fig


def plot_realized_volatility(
    df: pd.DataFrame,
    vol_col: str = "realized_vol_21d",
    title: str = "SPY 21-Day Realized Volatility",
) -> plt.Figure:
    """Plot the realized volatility time series."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[vol_col], linewidth=0.8, color="firebrick")
    ax.set_ylabel("Realized Volatility")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def plot_qq(
    series: pd.Series | np.ndarray,
    name: str = "Log Returns",
) -> plt.Figure:
    """Q-Q plot vs. normal distribution with reference line."""
    _apply_style()
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]

    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(arr, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue", label="data")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, slope * line_x + intercept, color="firebrick", linewidth=1.5, label="normal")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"Q-Q Plot: {name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_return_distribution(
    series: pd.Series | np.ndarray,
    name: str = "Log Returns",
    bins: int = 100,
) -> plt.Figure:
    """Histogram + KDE overlaid with a fitted normal curve."""
    _apply_style()
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(arr, bins=bins, stat="density", ax=ax, color="steelblue", alpha=0.5, label="Empirical")

    mu, sigma = arr.mean(), arr.std()
    x = np.linspace(arr.min(), arr.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="firebrick", linewidth=1.5, label="Normal fit")

    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    title: str = "Feature Correlation Matrix",
) -> plt.Figure:
    """Heatmap of the Pearson correlation matrix for *feature_cols*."""
    _apply_style()
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(feature_cols)), max(6, len(feature_cols) - 2)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_volatility_clustering(
    df: pd.DataFrame,
    return_col: str = "log_return",
    vol_col: str = "realized_vol_21d",
) -> plt.Figure:
    """
    Side-by-side: |log_return| over time (shows clustering) and
    ACF of squared returns (statistical signature of vol clustering).
    """
    from statsmodels.graphics.tsaplots import plot_acf

    _apply_style()
    arr = df[return_col].dropna().values
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(df.index, df[return_col].abs(), linewidth=0.5, color="dimgray", alpha=0.7)
    axes[0].set_title("|Log Return| — Volatility Clustering")
    axes[0].set_ylabel("|Log Return|")

    plot_acf(arr ** 2, lags=60, ax=axes[1], title="ACF of Squared Returns")
    axes[1].set_xlabel("Lag (days)")

    fig.tight_layout()
    return fig


# ── Misc helpers ──────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for notebook-friendly output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def describe_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Return a summary DataFrame of the three temporal splits."""
    rows = []
    for label, split in [("Train", train), ("Validation", val), ("Test", test)]:
        rows.append({
            "Split": label,
            "Start": split.index.min().date(),
            "End": split.index.max().date(),
            "Days": len(split),
        })
    return pd.DataFrame(rows).set_index("Split")
