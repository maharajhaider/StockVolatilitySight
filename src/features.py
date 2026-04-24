"""
Feature engineering and target variable construction.

Key transformations
-------------------
- Log returns, absolute returns, open-to-close return
- Intraday range (High - Low), log-scaled
- Log volume
- Rolling mean and std of log returns (windows: 5, 10, 21 days)
- Rolling mean of absolute returns (proxy for short-term vol)
- 21-day forward realized volatility (target variable):
      sigma_t = sqrt( (1/m) * sum_{j=1}^{m} (r_{t+j} - r_bar)^2 )
  per Andersen & Bollerslev (1998) adapted for daily data
- AAII investor-sentiment survey columns (bullish / bearish / neutral /
  bull_bear_spread), forward-filled to each trading day from weekly releases

All transforms produce NaN for any rows that cannot be computed (e.g. the
first row for log_return, or the last 20 rows for the target). The caller
is responsible for dropping NaNs before model training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import DATA_RAW, ROLLING_WINDOWS, SENTIMENT_FEATURES, VOL_WINDOW, TRADING_DAYS_YEAR

logger = logging.getLogger(__name__)


# ── Price-based features ───────────────────────────────────────────────────────

def add_log_return(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Daily log return: ln(P_t / P_{t-1})."""
    df = df.copy()
    df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))
    return df


def add_abs_return(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute value of log return — proxy for magnitude of daily move."""
    df = df.copy()
    if "log_return" not in df.columns:
        raise ValueError("Run add_log_return before add_abs_return.")
    df["abs_return"] = df["log_return"].abs()
    return df


def add_oc_return(df: pd.DataFrame, open_col: str = "Open", close_col: str = "Close") -> pd.DataFrame:
    """Open-to-close return: ln(Close / Open) — intraday directional move."""
    df = df.copy()
    df["oc_return"] = np.log(df[close_col] / df[open_col])
    return df


def add_intraday_range(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """
    Normalised intraday range: log( (High - Low) / Close ).
    Provides a scale-invariant measure of within-day price spread.
    """
    df = df.copy()
    spread = df[high_col] - df[low_col]
    # Guard against zero spread (rare but possible on halt days)
    spread = spread.replace(0, np.nan)
    df["intraday_range"] = np.log(spread / df[close_col])
    return df


def add_log_volume(df: pd.DataFrame, volume_col: str = "Volume") -> pd.DataFrame:
    """Log volume: log(1 + Volume) — stabilises the heavy right tail."""
    df = df.copy()
    df["log_volume"] = np.log1p(df[volume_col])
    return df


def add_relative_volume(df: pd.DataFrame, volume_col: str = "Volume", window: int = 21) -> pd.DataFrame:
    """Relative volume: Volume / 21-day rolling mean Volume."""
    df = df.copy()
    # Replace zeros to avoid division by zero
    v = df[volume_col].replace(0, np.nan)
    df[f"relative_volume_{window}d"] = v / v.rolling(window, min_periods=window).mean()
    return df


# ── Rolling features ───────────────────────────────────────────────────────────

def add_rolling_stats(
    df: pd.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling mean and std of log_return, and rolling mean of abs_return,
    for each window size in *windows*.

    Produces columns like:
      rolling_mean_5, rolling_std_5,
      rolling_mean_10, rolling_std_10,
      rolling_mean_21, rolling_std_21,
      rolling_abs_mean_5, ...
    """
    df = df.copy()
    if "log_return" not in df.columns:
        raise ValueError("Run add_log_return before add_rolling_stats.")

    for w in windows:
        df[f"rolling_mean_{w}"] = df["log_return"].rolling(w, min_periods=w).mean()
        df[f"rolling_std_{w}"] = df["log_return"].rolling(w, min_periods=w).std()
        df[f"rolling_abs_mean_{w}"] = df["abs_return"].rolling(w, min_periods=w).mean()

    return df


# ── Target: realized volatility ────────────────────────────────────────────────

def add_realized_volatility(
    df: pd.DataFrame,
    window: int = VOL_WINDOW,
    annualise: bool = False,
) -> pd.DataFrame:
    """
    Compute the *forward-looking* 21-day realized volatility target.

    Definition (Andersen & Bollerslev 1998, adapted for daily data):
        sigma_t = sqrt( (1/m) * sum_{j=1}^{m} (r_{t+j} - r_bar_{t+j})^2 )

    where r_bar_{t+j} is the mean return over the same forward window.
    This demeaned version is appropriate for lower-frequency (daily) data
    where the mean is non-negligible compared to the variance.

    The target is *forward-looking*: at time t, sigma_t aggregates returns
    over [t+1, t+m]. This means the last `window` rows will have NaN targets.

    Parameters
    ----------
    df        : DataFrame with 'log_return' column
    window    : rolling window in trading days (default 21)
    annualise : if True, multiply by sqrt(252) to express in annual terms

    Returns
    -------
    DataFrame with new column 'realized_vol_{window}d'
    """
    df = df.copy()
    if "log_return" not in df.columns:
        raise ValueError("Run add_log_return before add_realized_volatility.")

    target_col = f"realized_vol_{window}d"
    r = df["log_return"].values
    n = len(r)
    vol = np.full(n, np.nan)

    for t in range(n - window):
        window_returns = r[t + 1 : t + 1 + window]
        if np.any(np.isnan(window_returns)):
            continue
        mean_r = window_returns.mean()
        demeaned_sq = (window_returns - mean_r) ** 2
        vol[t] = np.sqrt(demeaned_sq.mean())

    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS_YEAR)

    df[target_col] = vol
    return df


# ── Chronological train/val/test split ────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_end: str = "2015-12-31",
    val_end: str = "2019-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df chronologically into train, validation, and test sets.

    No shuffling — respects temporal order to prevent leakage.

    Returns (train_df, val_df, test_df).
    """
    train = df.loc[: train_end].copy()
    val = df.loc[train_end + " ":val_end].copy()  # exclusive start via string slicing
    test = df.loc[val_end + " ":].copy()

    # Fix: pandas string slicing is inclusive; shift by one day
    train = df[df.index <= train_end].copy()
    val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df[df.index > val_end].copy()

    logger.info(
        "Split sizes — Train: %d, Val: %d, Test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ── Feature selection helpers ─────────────────────────────────────────────────

def drop_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Return a list of feature columns with highly correlated pairs removed.
    When two features correlate above *threshold*, the one that appears later
    in *feature_cols* is dropped.

    Does not modify *df* — returns the kept column names only.
    """
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info("Dropping correlated features (threshold=%.2f): %s", threshold, to_drop)

    return [c for c in feature_cols if c not in to_drop]


def variance_inflation_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    vif_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for *feature_cols*.
    Returns a DataFrame of (feature, VIF) pairs sorted descending.
    Requires statsmodels.

    High VIF (> 10) indicates multicollinearity worth investigating.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        sub = df[feature_cols].dropna()
        vif_data = pd.DataFrame({
            "feature": feature_cols,
            "VIF": [
                variance_inflation_factor(sub.values, i)
                for i in range(len(feature_cols))
            ],
        }).sort_values("VIF", ascending=False)

        flagged = vif_data[vif_data["VIF"] > vif_threshold]
        if not flagged.empty:
            logger.warning(
                "Features with VIF > %.1f (potential multicollinearity):\n%s",
                vif_threshold,
                flagged.to_string(index=False),
            )
        return vif_data
    except ImportError:
        logger.warning("statsmodels not installed — skipping VIF check.")
        return pd.DataFrame(columns=["feature", "VIF"])


# ── Sentiment (AAII) ───────────────────────────────────────────────────────────

def ensure_sentiment_columns(
    df: pd.DataFrame,
    raw_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Guarantee ``SENTIMENT_FEATURES`` on *df*, aligned to its index.

    If columns are missing (e.g. raw merge skipped the file), load from *raw_dir*
    (default ``config.DATA_RAW``) using ``load_aaii_sentiment`` (``.csv`` / ``.xlsx`` /
    ``.xls``), forward-fill weekly releases to daily rows, then join. Raises
    ``FileNotFoundError`` if no supported file is present.
    """
    from src.data_loader import _forward_fill_weekly_to_daily, load_aaii_sentiment

    rd = Path(raw_dir) if raw_dir is not None else DATA_RAW
    need_load = any(c not in df.columns for c in SENTIMENT_FEATURES)
    if need_load:
        sent = load_aaii_sentiment(raw_dir=rd)
        if sent.empty:
            raise FileNotFoundError(
                "AAII sentiment is required for the feature matrix. Download weekly "
                "survey results from https://www.aaii.com/sentimentsurvey/sent_results "
                f"and save as {rd / 'aaii_sentiment.csv'} (or .xlsx / .xls)"
            )
        aligned = _forward_fill_weekly_to_daily(sent, df.index)
        for c in SENTIMENT_FEATURES:
            if c not in aligned.columns:
                raise ValueError(f"Sentiment loader missing expected column {c!r}")
            df[c] = aligned[c].reindex(df.index).to_numpy()
    for c in SENTIMENT_FEATURES:
        df[c] = df[c].ffill()
    return df


# ── Master pipeline ────────────────────────────────────────────────────────────

def build_features(
    raw_df: pd.DataFrame,
    vol_window: int = VOL_WINDOW,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
    annualise_target: bool = False,
    raw_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps to *raw_df* in sequence.

    Steps applied
    -------------
    1.  Log return
    2.  Absolute return
    3.  Open-to-close return
    4.  Intraday range (log-normalised)
    5.  Log volume
    6.  Rolling mean / std of log_return (multiple windows)
    7.  Forward realized volatility target
    8.  AAII sentiment columns (weekly → daily forward fill)

    NaN rows at the start (due to lagging) and end (due to forward target)
    are NOT dropped here — the caller decides when to drop them.

    Returns the enriched DataFrame with all original columns preserved.
    """
    df = raw_df.copy()

    df = add_log_return(df)
    df = add_abs_return(df)
    df = add_oc_return(df)
    df = add_intraday_range(df)
    df = add_log_volume(df)
    df = add_relative_volume(df)
    df = add_rolling_stats(df, windows=rolling_windows)
    df = add_realized_volatility(df, window=vol_window, annualise=annualise_target)
    df = ensure_sentiment_columns(df, raw_dir=raw_dir)

    target_col = f"realized_vol_{vol_window}d"
    n_nan_start = df["log_return"].isna().sum()
    n_nan_end = df[target_col].isna().sum() - n_nan_start
    logger.info(
        "Feature build complete. Rows with NaN: %d leading (lag), ~%d trailing (forward target).",
        n_nan_start,
        n_nan_end,
    )
    return df
