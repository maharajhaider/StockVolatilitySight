---
name: HMM-LSTM Volatility Forecast
overview: Implement a regime-aware HMM-LSTM hybrid model for SPY volatility prediction, including data engineering, HMM regime detection with model selection diagnostics, regime-specific LSTM training, ensemble prediction, and evaluation against a baseline LSTM.
todos:
  - id: phase1-data
    content: "Phase 1: Data collection (yfinance SPY, put/call ratio, AAII sentiment), feature engineering (log returns, ranges, rolling stats), target variable (21-day realized vol), chronological split, normality diagnostics"
    status: completed
  - id: phase2-hmm
    content: "Phase 2: HMM regime detection -- train GaussianHMM vs GMMHMM, compare BIC/log-likelihood; test 1st vs 2nd order Markov; Viterbi regime assignment with sanity-check plots"
    status: completed
  - id: phase3-lstm
    content: "Phase 3: LSTM training -- baseline LSTM on full data (3a DONE: PyTorch CPU + Optuna TPE, log-target, inverse-log MSE), then regime-specific LSTMs (calm + volatile) on HMM-split data"
    status: completed
  - id: phase4-ensemble
    content: "Phase 4: Ensemble prediction -- combine regime LSTMs weighted by HMM soft probabilities"
    status: completed
  - id: phase5-eval
    content: "Phase 5: Evaluation -- metrics and plots comparing ensemble vs baseline"
    status: completed
isProject: false
---

# HMM-LSTM Stock Volatility Prediction Implementation

## Plan Updates

### Phase 4 & Phase 5 Complete - branch `phase3-resplit`

- `src/ensemble.py` aggregates and joins predictions from `lstm_baseline`, `lstm_calm`, and `lstm_volatile` using Pandas timestamp alignment.
- A naïve baseline prediction (current day's historical `rolling_std_21`) is included.
- `notebooks/06_ensemble_eval.ipynb` performs plotting, metrics generation and regime-specific breakdown.
- **Results**: The ensemble logic (MSE: 0.00167) soundly beats the baseline model (MSE: 0.00246) across the holdout 2020-present test set!

---

## Scope Decisions

- **Stock**: SPY only (20+ years of daily data available via Yahoo Finance)
- Drop XEQT (insufficient history) and MAG7 individual stocks from scope
- **Target**: 21-day rolling realized volatility (as defined in the proposal)
- **Chronological split**: Train 2004-2015, Validation 2016-2019, Test 2020-present

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

### 3a. Baseline LSTM (no regime separation) — DONE

- **Framework**: PyTorch, CPU-only (`torch` installed in `venv/`, `cuda=False`)
- **Input**: configurable feature set (default: `Open, High, Low, Close, Volume, log_return, abs_return, oc_return, intraday_range, log_volume`) over a sliding window; `seq_len` is itself a tuned hyperparameter
- **Output**: log of 21-day forward realized volatility; predictions are `exp()`-ed back before metric computation
- **Target transform**: `log(realized_vol_21d)` trained with MSE loss; validation/test MSE is computed in the *original* (inverse-log) volatility scale so the scoring is invariant to the transform
- **Feature scaling**: `StandardScaler` fit on the train split, applied to val/test
- **Architecture**: stacked `nn.LSTM` → `nn.Linear(hidden_size, 1)`; final timestep used as the regression summary
- **Hyperparameter tuning**: **Optuna** (TPE sampler) over `LSTM_SEARCH_SPACE` in `config.py` — `hidden_size`, `n_layers`, `dropout`, `lr`, `batch_size`, `seq_len` — scored by validation MSE (raw scale); `LSTM_N_TRIALS` controls the budget
- **Final pass**: retrain with Optuna's best params on the full training set, early-stopping on val, then evaluate on test
- **Artifacts**: `models/lstm_baseline.pt` (state_dict + hyperparameters + feature list), `models/lstm_baseline_scaler.joblib`
- **Entry points**: `src/train_LSTM_baseline.py` (CLI: `--features`, `--n-trials`, `--tune-epochs`, `--final-epochs`, ...) and the runner notebook `notebooks/04_lstm_baseline.ipynb`
- **Results** (20 trials): best params `hidden_size=128, n_layers=1, dropout=0.155, lr=4.5e-4, seq_len=21`; test MSE=7.6e-4, RMSE=0.0276, MAE=0.0244 (1,538 windows)

### 3b. Regime-Specific LSTMs — DONE

- **Implementation**: `src/train_LSTM_regime.py` (new); runner notebook `notebooks/05_lstm_regime.ipynb` (new)
- **Window assignment**: `RegimeWindowDataset` filters sliding windows by dominant Viterbi state (majority vote across `seq_len` timesteps); windows spanning both regimes go to the majority regime
- **Val fallback** (deviation from plan): the 2016-2019 val period had 0 volatile-dominant windows, so the volatile LSTM falls back to the full unfiltered val loader for early-stopping when there are 0 regime-filtered val windows
- **Calm LSTM** — 2,819 training windows; best params tuned on regime-filtered val; artifacts `models/lstm_calm.pt` + `models/lstm_calm_scaler.joblib`
- **Volatile LSTM** — 181 training windows (GFC/COVID/rate-hike periods); val fallback used (0 volatile val windows in 2016-2019); best params `hidden_size=32, n_layers=3, dropout=0.42, lr=9.1e-3, seq_len=42`; test MSE=4.9e-5, RMSE=0.0070, MAE=0.0055; artifacts `models/lstm_volatile.pt` + `models/lstm_volatile_scaler.joblib`
- **Test evaluation**: both regime LSTMs are evaluated on the **full** test set (not filtered) so the ensemble can call them on every window

**Deliverable:** Three trained LSTM models (baseline, calm, volatile) saved as checkpoints. ✅ Complete on branch `phase3-lstm`.

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
    lstm_model.py         # LSTM architecture + sliding-window Dataset
    train_LSTM_baseline.py # Phase 3a end-to-end: load → tune (Optuna) → retrain → test
    ensemble.py           # weighted prediction logic
    utils.py              # plotting, metrics, common helpers
  models/                 # saved model checkpoints
  config.py               # hyperparams, date splits, feature lists
```

## Key Dependencies

- `yfinance`, `pandas`, `numpy`, `scipy` (normality tests)
- `hmmlearn` (GaussianHMM, GMMHMM)
- `torch` (LSTM, CPU build)
- `optuna` (TPE hyperparameter search)
- `scikit-learn` (scaling, metrics, optional tree baseline)
- `matplotlib`, `seaborn` (plots)
- `joblib` (model serialization)
