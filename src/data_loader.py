"""
Data loading utilities.

Responsibilities:
- Download SPY OHLCV via yfinance
- Load CBOE put/call ratio (manual CSV or pandas_datareader)
- Load and align AAII weekly sentiment survey data
- Merge all sources into a single daily DataFrame indexed by date
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# ── SPY OHLCV ─────────────────────────────────────────────────────────────────

def download_spy(
    ticker: str = "SPY",
    start: str = "2004-01-01",
    end: str | None = None,
    raw_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for *ticker* using yfinance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Adjusted close is used for Close (accounts for splits/dividends).

    Parameters
    ----------
    ticker      : Yahoo Finance ticker symbol
    start       : ISO date string, inclusive
    end         : ISO date string, exclusive; None → today
    raw_dir     : directory to cache the raw parquet; None → no caching
    force_refresh : ignore cache and re-download
    """
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        cache_path = raw_dir / f"{ticker}_ohlcv.parquet"
        if cache_path.exists() and not force_refresh:
            logger.info("Loading SPY OHLCV from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    logger.info("Downloading %s OHLCV from Yahoo Finance (%s → %s)…", ticker, start, end or "today")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached SPY OHLCV → %s", cache_path)

    return df


# ── CBOE Put/Call Ratio ────────────────────────────────────────────────────────

def load_putcall_ratio(
    csv_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> pd.Series:
    """
    Load the CBOE total put/call ratio.

    Priority:
    1. If *csv_path* is given, read that file directly.
    2. Otherwise look for 'put_call_ratio.csv' in *raw_dir*.
    3. If neither exists, return an empty Series with a note — the notebook
       will guide the user to download the file manually from
       https://www.cboe.com/us/options/market_statistics/daily/

    Expected CSV columns (CBOE standard): DATE, P/C Ratio  (or similar)
    Returns a Series named 'put_call_ratio', indexed by date.
    """
    if csv_path is None and raw_dir is not None:
        candidate = Path(raw_dir) / "put_call_ratio.csv"
        if candidate.exists():
            csv_path = candidate

    if csv_path is None or not Path(csv_path).exists():
        logger.warning(
            "Put/call ratio CSV not found. "
            "Download from https://www.cboe.com/us/options/market_statistics/daily/ "
            "and save as data/raw/put_call_ratio.csv"
        )
        return pd.Series(name="put_call_ratio", dtype=float)

    df = pd.read_csv(csv_path, parse_dates=["DATE"])
    # CBOE files use 'DATE' and 'P/C Ratio' or 'TOTAL PUT/CALL RATIO'
    date_col = "DATE"
    ratio_col = next(
        (c for c in df.columns if "ratio" in c.lower() or "p/c" in c.lower()),
        df.columns[1],
    )
    series = df.set_index(date_col)[ratio_col].rename("put_call_ratio")
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    # Replace erroneous zeros with NaN
    series = series.replace(0, np.nan)
    return series


# ── AAII Sentiment ─────────────────────────────────────────────────────────────

def load_aaii_sentiment(
    csv_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load AAII weekly sentiment survey data.

    Source: https://www.aaii.com/sentimentsurvey/sent_results
    Download the export from AAII (often **Excel** `.xls` / `.xlsx`, sometimes CSV) and
    save under ``data/raw/`` as ``aaii_sentiment.csv`` **or** ``aaii_sentiment.xlsx`` /
    ``aaii_sentiment.xls`` (any of these names is detected automatically).

    Expected columns: Date, Bullish, Neutral, Bearish (percentages as decimals or %)
    Returns a DataFrame with columns: bullish, neutral, bearish, bull_bear_spread,
    indexed by date (forward-filled to daily frequency for merging).
    """
    if csv_path is None and raw_dir is not None:
        rd = Path(raw_dir)
        for name in (
            "aaii_sentiment.csv",
            "aaii_sentiment.xlsx",
            "aaii_sentiment.xls",
        ):
            candidate = rd / name
            if candidate.exists():
                csv_path = candidate
                break

    if csv_path is None or not Path(csv_path).exists():
        logger.warning(
            "AAII sentiment file not found. "
            "Download from https://www.aaii.com/sentimentsurvey/sent_results "
            "and save as data/raw/aaii_sentiment.csv (or .xlsx / .xls)"
        )
        return pd.DataFrame(columns=["bullish", "neutral", "bearish", "bull_bear_spread"])

    path = Path(csv_path)
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        # Both the .xls and .xlsx exports from AAII share the same 5-row preamble:
        #   row 0, 1 – blank / merged organisation banner
        #   row 2    – "Reported" + span labels
        #   row 3    – real column names: Date, Bullish, Neutral, Bearish …  ← header
        #   row 4    – blank separator
        #   row 5+   – weekly data (1987-06-26 …)
        # Skip rows 0-2 (junk) and row 4 (blank separator); use row 3 as the header.
        # read_excel auto-parses Excel date cells; passing dtype={"date": object} is
        # not possible before we know the column name, so we also coerce below.
        df = pd.read_excel(path, skiprows=[0, 1, 2, 4])
    else:
        df = pd.read_csv(path)

    # Normalize column names to lowercase, strip whitespace
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Identify date column
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    # Coerce unparseable values (e.g. footer summary rows like "Reported") to NaT,
    # then drop them.  This guards against any remaining junk rows for all formats.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # Identify percentage columns
    def _find_col(keywords: list[str]) -> str | None:
        for kw in keywords:
            for c in df.columns:
                if kw in c:
                    return c
        return None

    bull_col = _find_col(["bull"])
    bear_col = _find_col(["bear"])
    neut_col = _find_col(["neut"])

    out = pd.DataFrame(index=df.index)
    for col, name in [(bull_col, "bullish"), (bear_col, "bearish"), (neut_col, "neutral")]:
        if col is not None:
            vals = df[col].copy()
            # Convert from percentage (e.g. 45.0) to decimal (0.45) if needed
            if vals.dropna().max() > 1.5:
                vals = vals / 100.0
            out[name] = vals
        else:
            out[name] = np.nan

    out["bull_bear_spread"] = out["bullish"] - out["bearish"]
    return out


def _forward_fill_weekly_to_daily(
    weekly: pd.DataFrame | pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame | pd.Series:
    """
    Reindex a weekly series/DataFrame to a daily index using forward fill.
    Sentiment is released weekly; we carry it forward until the next release.
    """
    combined_index = daily_index.union(weekly.index).sort_values()
    reindexed = weekly.reindex(combined_index).ffill()
    return reindexed.reindex(daily_index)


# ── Master merge ──────────────────────────────────────────────────────────────

def build_raw_dataset(
    ticker: str = "SPY",
    start: str = "2004-01-01",
    end: str | None = None,
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build the master raw DataFrame by joining SPY OHLCV, put/call ratio,
    and AAII sentiment on a common daily trading-day index.

    Missing secondary sources produce NaN columns rather than raising errors,
    so the pipeline can proceed with partial data.

    Parameters
    ----------
    ticker        : equity ticker (default 'SPY')
    start         : start date for download
    end           : end date for download (None → today)
    raw_dir       : directory containing supplementary CSVs and OHLCV cache
    processed_dir : if provided, save the merged DataFrame as a parquet here
    force_refresh : force re-download even if cache exists

    Returns
    -------
    pd.DataFrame indexed by Date with all available raw columns.
    """
    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        out_path = processed_dir / "raw_dataset.parquet"
        if out_path.exists() and not force_refresh:
            logger.info("Loading merged raw dataset from cache: %s", out_path)
            return pd.read_parquet(out_path)

    # 1. SPY OHLCV
    ohlcv = download_spy(ticker, start=start, end=end, raw_dir=raw_dir, force_refresh=force_refresh)

    # 2. Put/call ratio (daily)
    pc = load_putcall_ratio(raw_dir=raw_dir)
    if not pc.empty:
        pc = pc.reindex(ohlcv.index)  # align to trading days, NaN for missing

    # 3. AAII sentiment (weekly → daily)
    sentiment = load_aaii_sentiment(raw_dir=raw_dir)
    if not sentiment.empty:
        sentiment = _forward_fill_weekly_to_daily(sentiment, ohlcv.index)

    # 4. Merge
    df = ohlcv.copy()
    if not pc.empty:
        df = df.join(pc, how="left")
    if not sentiment.empty:
        df = df.join(sentiment, how="left")

    # 5. Report missingness
    missing_pct = df.isnull().mean() * 100
    if missing_pct.any():
        logger.info("Missing data per column (%%): \n%s", missing_pct[missing_pct > 0].to_string())

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info("Saved merged raw dataset → %s", out_path)

    return df
