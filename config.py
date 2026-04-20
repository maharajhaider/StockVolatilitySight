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
START_DATE = "2001-04-01"  # extra year so 21-day vol target is available from 2005
END_DATE = None  # None → today

# Chronological split boundaries
TRAIN_END = "2020-12-31"
VAL_END = "2023-04-30"
# Test: 2020-01-01 → present

# ── Target ────────────────────────────────────────────────────────────────────
VOL_WINDOW = 21  # rolling window (trading days) for realized volatility
TRADING_DAYS_YEAR = 252  # used for annualisation if needed

# ── Feature groups ────────────────────────────────────────────────────────────
OHLCV_FEATURES = [
    "log_return",
    "abs_return",
    "intraday_range",
    "log_volume",
    "oc_return",  # open-to-close return
]

ROLLING_WINDOWS = [5, 10, 21]  # windows for rolling mean/std features

SENTIMENT_FEATURES = [
    "bullish",
    "bearish",
    "neutral",
    "bull_bear_spread",
]

PUTCALL_FEATURES = [
    "put_call_ratio",
]

# Features used as HMM input.
# Source: notebook 01 drop_correlated_features(threshold=0.95) output.
# rolling_abs_mean_{5,10,21} were the only features dropped (r > 0.95 with
# rolling_std_* counterparts); the remaining 11 features are kept as-is.
# The VIF check in notebook 01 is informational only and removes nothing.
HMM_FEATURES = [
    "log_return",
    "abs_return",
    "oc_return",
    "intraday_range",
    "log_volume",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_21",
    "rolling_std_21",
]

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES = 2
HMM_N_MIX = [1, 2, 3]  # n_mix values to compare (1 = GaussianHMM)
HMM_MARKOV_ORDERS = [1, 2]  # orders to compare
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 200
HMM_RANDOM_STATE = 42
HMM_N_RESTARTS = 50          # random restarts per model candidate

# Penalised EM: λ in penalised_LL = log P(O|model) − λ · Σ_s(1 − p_stay_s).
# Discourages degenerate rapid-switching solutions during restart selection.
# Rule of thumb: set ≈ number of training observations (~3,000 for 2004–2015).
# Reference: Nystrup, Hansen & Madsen (2015–2019).
HMM_DURATION_PENALTY = 3000.0

# ── LSTM ──────────────────────────────────────────────────────────────────────
# Log-transform the target: realized vol is right-skewed (skew=3.5, kurt=17.4).
# log(vol) reduces skew to 0.75 and kurt to 1.07 — much better for MSE training.
# Predictions must be exponentiated back: y_pred = exp(model_output)
LOG_TRANSFORM_TARGET = True

LSTM_PATIENCE = 10  # early stopping patience
LSTM_RANDOM_STATE = 42

# Default input feature columns for the baseline LSTM.
# Can be overridden on the command line via --features.
LSTM_BASELINE_FEATURES = [
    "log_return",
    "abs_return",
    "oc_return",
    "intraday_range",
    "relative_volume_21d",
]

# Target column produced by features.build_features().
LSTM_TARGET = f"realized_vol_{VOL_WINDOW}d"

# ── Optuna search space for the baseline LSTM ─────────────────────────────────
# Categorical lists are passed to trial.suggest_categorical; (low, high) tuples
# are passed to trial.suggest_float (log scale for lr) or suggest_int.
LSTM_N_TRIALS = 20
LSTM_TUNE_EPOCHS = 40      # epochs per trial during tuning (shorter than final)
LSTM_FINAL_EPOCHS = 100    # epochs for the final retrain with best params

LSTM_SEARCH_SPACE = {
    "hidden_size": [32, 64, 128],
    "n_layers": [1, 2, 3],
    "dropout": (0.0, 0.5),       # continuous uniform
    "lr": (1e-4, 1e-2),           # log-uniform
    "batch_size": [32, 64, 128],
    "seq_len": [21, 42],
}

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
NORMALITY_ALPHA = 0.05  # significance level for normality tests
