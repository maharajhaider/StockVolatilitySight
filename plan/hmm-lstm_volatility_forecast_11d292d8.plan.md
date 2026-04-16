---
name: HMM-LSTM Volatility Forecast
overview: Implement a regime-aware HMM-LSTM hybrid model for SPY volatility prediction, including data engineering, HMM regime detection with model selection diagnostics, regime-specific LSTM training, ensemble prediction, and evaluation against a baseline LSTM.
todos:
  - id: phase1-data
    content: "Phase 1: Data collection (yfinance SPY, put/call ratio, AAII sentiment), feature engineering (log returns, ranges, rolling stats), target variable (21-day realized vol), chronological split, normality diagnostics"
    status: pending
  - id: phase2-hmm
    content: "Phase 2: HMM regime detection -- train GaussianHMM vs GMMHMM, compare BIC/log-likelihood; test 1st vs 2nd order Markov; Viterbi regime assignment with sanity-check plots"
    status: pending
  - id: phase3-lstm
    content: "Phase 3: LSTM training -- baseline LSTM on full data, then regime-specific LSTMs (calm + volatile) on HMM-split data; hyperparameter tuning on validation set"
    status: pending
  - id: phase4-ensemble
    content: "Phase 4: Ensemble prediction -- combine regime LSTMs weighted by HMM soft probabilities"
    status: pending
  - id: phase5-eval
    content: "Phase 5: Evaluation -- MSE/RMSE/MAE comparison of ensemble vs baseline; predicted-vs-actual plots, error analysis, per-regime breakdown"
    status: pending
isProject: false
---

# HMM-LSTM Stock Volatility Prediction Implementation

## Scope Decisions

- **Stock**: SPY only (20+ years of daily data available via Yahoo Finance)
- Drop XEQT (insufficient history) and MAG7 individual stocks from scope
- **Target**: 21-day rolling realized volatility (as defined in the proposal)
- **Chronological split**: Train 2005-2015, Validation 2015-2020, Test 2020-present

---

## Phase 1: Data Collection and Engineering

**Data sources:**

- SPY daily OHLCV via `yfinance` (2005 -- present)
- CBOE Put/Call ratio (free CSV from CBOE website or `pandas_datareader`)
- AAII Weekly Sentiment Survey (downloadable CSV from aaii.com)

**Feature engineering:**

- Log returns: `log(Close_t / Close_{t-1})`
- Absolute returns
- Intraday range: `High - Low` (and log-scaled)
- Log volume: `log(Volume)`
- Rolling stats: 5/10/21-day rolling mean and std of returns
- Target variable: 21-day forward rolling realized volatility using the formula from the proposal:
  `sigma_t = sqrt(sum_{j=1}^{m} (r_{t+j} - r_bar)^2)` where `m=21`

**Data cleaning:**

- Handle missing trading days, NaN forward-fill for sentiment (weekly to daily alignment)
- Drop or impute any gaps in OHLCV
- Basic correlation check to prune redundant features (correlation matrix, VIF)

**Normality diagnostics (feeds into Phase 2):**

- Shapiro-Wilk test on log returns and feature distributions
- Q-Q plots
- Skewness and kurtosis statistics
- These results directly inform whether GaussianHMM or GMMHMM is appropriate

**Deliverable:** A single cleaned DataFrame saved as parquet, plus a notebook documenting EDA and normality results.

---

## Phase 2: HMM Regime Detection

This is the key model-selection phase with three diagnostic questions to answer:

### 2a. Simple Gaussian vs Gaussian Mixture

- Train `GaussianHMM(n_components=2)` via `hmmlearn`
- Train `GMMHMM(n_components=2, n_mix=2)` (and try `n_mix=3`)
- Compare using:
  - Log-likelihood on validation set
  - BIC (lower is better): `BIC = -2 * log_likelihood + k * log(n)`
  - AIC for additional reference
- Pick the winner; if GMMHMM wins, use it going forward

### 2b. Higher-Order Markov (1st vs 2nd order)

`hmmlearn` only supports first-order HMMs natively. To test second-order:

- Augment the state space: create composite states `(s_{t-1}, s_t)` giving 4 hidden states for a 2-state model
- Train `GaussianHMM(n_components=4)` (or GMMHMM with 4) on the same features
- Compare BIC/log-likelihood of 2-state vs 4-state (augmented 2nd-order)
- If 4-state model is not materially better on validation, stick with 1st-order (simpler)

### 2c. Regime Assignment and Sanity Check

- Run Viterbi decoding on full dataset to get most-likely regime sequence
- Plot regime overlay on SPY price chart -- verify that "volatile" state aligns with known crises (2008 GFC, 2020 COVID, 2022 rate hikes)
- Use `predict_proba()` (forward-backward) to get soft regime probabilities for ensemble step

**Deliverable:** A notebook with model comparison tables, regime overlay plots, and a final trained HMM saved via `joblib`.

---

## Phase 3: LSTM Training

### 3a. Baseline LSTM (no regime separation)

- Input: 21-day sliding window of all features
- Output: predicted 21-day realized volatility
- Architecture: 1-2 LSTM layers (64-128 units), dropout, dense output
- Train on full training set (2005-2015), tune hyperparameters on validation (2015-2020)
- Framework: PyTorch (or Keras -- user preference)

### 3b. Regime-Specific LSTMs

- Split training data by Viterbi-decoded regime labels
- Train `LSTM_calm` on calm-regime windows only
- Train `LSTM_volatile` on volatile-regime windows only
- Same architecture as baseline, but each sees only its regime's data
- Handle regime transitions at window boundaries (a window spanning both regimes gets assigned to the dominant regime)

**Hyperparameter tuning** (on validation set):

- Hidden size, number of layers, dropout rate, learning rate, batch size
- Use simple grid or random search

**Deliverable:** Three trained LSTM models (baseline, calm, volatile) saved as checkpoints.

---

## Phase 4: Ensemble Prediction

- For each test window, get regime probabilities from HMM: `P(calm | observations)`, `P(volatile | observations)`
- Get predictions from both regime LSTMs: `y_calm`, `y_volatile`
- Final prediction: `y_final = P(calm) * y_calm + P(volatile) * y_volatile`

---

## Phase 5: Evaluation

**Metrics** (on test set 2020-present):

- MSE, RMSE, MAE
- Compare: Ensemble HMM-LSTM vs Baseline LSTM
- Optional simple baseline: rolling historical volatility (naive), or a basic tree regressor

**Visualization:**

- Predicted vs actual volatility time series overlay
- Error distribution plots
- Per-regime performance breakdown

---

## Project Structure

```
StockVolatilitySight/
  README.md
  requirements.txt
  data/
    raw/                  # downloaded CSVs
    processed/            # cleaned parquet files
  notebooks/
    01_data_collection.ipynb
    02_eda_normality.ipynb
    03_hmm_regime.ipynb
    04_lstm_training.ipynb
    05_ensemble_eval.ipynb
  src/
    data_loader.py        # yfinance download, sentiment loading
    features.py           # feature engineering, target calculation
    hmm_model.py          # HMM training, comparison, Viterbi
    lstm_model.py         # LSTM architecture, train/eval loops
    ensemble.py           # weighted prediction logic
    utils.py              # plotting, metrics, common helpers
  models/                 # saved model checkpoints
  config.py               # hyperparams, date splits, feature lists
```

## Key Dependencies

- `yfinance`, `pandas`, `numpy`, `scipy` (normality tests)
- `hmmlearn` (GaussianHMM, GMMHMM)
- `torch` (LSTM)
- `scikit-learn` (scaling, metrics, optional tree baseline)
- `matplotlib`, `seaborn` (plots)
- `joblib` (model serialization)
