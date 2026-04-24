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
TRAIN_END = "2015-12-31"
VAL_END = "2020-04-30"
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

# neutral  excluded: = 1 - bullish - bearish (exact linear combination).
# bull_bear_spread excluded: = bullish - bearish (exact linear combination).
# Keeping only bullish + bearish gives the sentiment signal without
# redundant features that add noise and cause rank-deficient covariances in HMM.
SENTIMENT_FEATURES = [
    "bullish",
    "bearish",
]

PUTCALL_FEATURES = [
    "put_call_ratio",
]

# VIX-family features used in the forward-looking feature experiment (variant B).
# - vix                         : raw VIX closing level (bounded, mean-reverting)
# - vix_log_change              : ln(VIX_t / VIX_{t-1}) — stationary daily IV shock
# - vix3m_minus_vix             : 3-month − spot VIX — term-structure slope
#
# Source: FRED (VIXCLS, VXVCLS). SKEW and VVIX were dropped because FRED does
# not host them and yfinance's equivalents (^SKEW, ^VVIX) are too unreliable
# to depend on. VIX3M starts 2007-12-04 on FRED, so variant-B training
# effectively starts 2007-12 (~150 fewer train rows than variant A's 2005-01).
VIX_FAMILY_FEATURES = [
    "vix",
    "vix_log_change",
    "vix3m_minus_vix",
]

# Price / volume HMM inputs (notebook 01 correlation prune; VIF informational only).
HMM_PRICE_VOLUME_FEATURES = [
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

# HMM input = stationary market features + bullish/bearish sentiment levels.
# neutral is excluded (= 1 - bullish - bearish, perfectly collinear → singular cov).
# bull_bear_spread is excluded (= bullish - bearish, linear combination of the two).
# Keeping only bullish + bearish breaks the linear dependency while still giving
# the HMM the sentiment signal without a rank-deficient covariance matrix.
HMM_FEATURES = HMM_PRICE_VOLUME_FEATURES + ["bullish", "bearish"]

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

# Stationary LSTM inputs without survey data (Phase 6); use for ablations.
LSTM_STATIONARY_FEATURES = [
    "log_return",
    "abs_return",
    "oc_return",
    "intraday_range",
    "relative_volume_21d",
]

# Default baseline LSTM = stationary + AAII sentiment (requires aaii_sentiment.csv/.xls in data/raw/).
LSTM_BASELINE_FEATURES = LSTM_STATIONARY_FEATURES + SENTIMENT_FEATURES

# ── Variant feature sets for the seven-model comparison experiment ────────────
# Variant A = current baseline (LSTM_BASELINE_FEATURES above — stationary + sentiment).
# Variant B = A augmented with VIX-family forward-looking features.
# Variant H = A augmented with the HMM posterior p_volatile as an additional
#             input feature. Tests whether the regime signal can be exploited
#             through a feature in a single LSTM instead of via the regime-split
#             ensemble architecture. Only p_volatile is added (p_calm is the
#             collinear complement; Ridge stacking in src/gate_stacking.py
#             already demonstrated both add the same information).
#
# HMM features are NOT augmented with VIX — regime labels remain derived from
# price/volume + sentiment only so that variant A and variant B share the same
# HMM regime sequence. This isolates the LSTM's response to the new features.
LSTM_VARIANT_A_FEATURES = list(LSTM_BASELINE_FEATURES)
LSTM_VARIANT_B_FEATURES = LSTM_VARIANT_A_FEATURES + VIX_FAMILY_FEATURES
LSTM_VARIANT_H_FEATURES = LSTM_VARIANT_A_FEATURES + ["p_volatile"]

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
