"""
Central configuration for StockVolatilitySight.
All hyperparameters, paths, and constants live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER = "SPY"
START_DATE = "2004-01-01"   # extra year so 21-day vol target is available from 2005
END_DATE = None             # None → today

# Chronological split boundaries
TRAIN_END = "2015-12-31"
VAL_END = "2019-12-31"
# Test: 2020-01-01 → present

# ── Target ────────────────────────────────────────────────────────────────────
VOL_WINDOW = 21             # rolling window (trading days) for realized volatility
TRADING_DAYS_YEAR = 252     # used for annualisation if needed

# ── Feature groups ────────────────────────────────────────────────────────────
OHLCV_FEATURES = [
    "log_return",
    "abs_return",
    "intraday_range",
    "log_volume",
    "oc_return",            # open-to-close return
]

ROLLING_WINDOWS = [5, 10, 21]   # windows for rolling mean/std features

SENTIMENT_FEATURES = [
    "bullish",
    "bearish",
    "neutral",
    "bull_bear_spread",
]

PUTCALL_FEATURES = [
    "put_call_ratio",
]

# Features used as HMM input (subset of all engineered features)
HMM_FEATURES = [
    "log_return",
    "abs_return",
    "intraday_range",
    "log_volume",
    "rolling_std_21",
]

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES = 2
HMM_N_MIX = [1, 2, 3]          # n_mix values to compare (1 = GaussianHMM)
HMM_MARKOV_ORDERS = [1, 2]      # orders to compare
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 200
HMM_RANDOM_STATE = 42

# ── LSTM ──────────────────────────────────────────────────────────────────────
# Log-transform the target: realized vol is right-skewed (skew=3.5, kurt=17.4).
# log(vol) reduces skew to 0.75 and kurt to 1.07 — much better for MSE training.
# Predictions must be exponentiated back: y_pred = exp(model_output)
LOG_TRANSFORM_TARGET = True

LSTM_SEQ_LEN = 21               # input sequence length (matches vol window)
LSTM_HIDDEN_SIZE = 64
LSTM_N_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_PATIENCE = 10              # early stopping patience
LSTM_RANDOM_STATE = 42

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
NORMALITY_ALPHA = 0.05          # significance level for normality tests
