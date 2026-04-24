# StockVolatilitySight

Regime-aware volatility prediction using a Hidden Markov Model + LSTM hybrid architecture on SPY.

## Overview

The model first uses an HMM to classify each trading day into a **calm** or **volatile** regime, then routes predictions through regime-specific LSTMs, combining outputs via HMM soft probabilities.

**Target variable:** 21-day forward realized volatility (Andersen & Bollerslev, 1998 adapted for daily data)

**Data:** SPY daily OHLCV (2004–present) via yfinance, with optional CBOE put/call ratio and AAII sentiment survey data.

## Setup

```bash
pip install -r requirements.txt
```

## Supplementary data (optional but recommended)

Download these manually and save to `data/raw/`:

| File | Source |
|------|--------|
| `put_call_ratio.csv` | https://www.cboe.com/us/options/market_statistics/daily/ |
| `aaii_sentiment.csv` (or `.xlsx` / `.xls` with the same base name) | https://www.aaii.com/sentimentsurvey/sent_results |

Put/call can be omitted (those columns stay NaN). The default model features include AAII sentiment, so you need one of the `aaii_sentiment.*` files above for `build_features` / training unless you override features to exclude sentiment.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_collection.ipynb` | Download data, engineer features, save splits |
| `02_eda_normality.ipynb` | Distribution analysis, normality tests, volatility clustering |
| `03_hmm_regime.ipynb` | HMM training, GaussianHMM vs GMMHMM comparison, Viterbi decoding |
| `04_lstm_training.ipynb` | Baseline LSTM + regime-specific LSTMs |
| `05_ensemble_eval.ipynb` | Weighted ensemble prediction, final evaluation |

## Project structure

```
StockVolatilitySight/
  config.py              Central hyperparameters and paths
  requirements.txt
  src/
    data_loader.py       yfinance download, put/call, AAII loading
    features.py          Feature engineering, target variable, train/val/test split
    utils.py             Plotting helpers, normality tests, regression metrics
    hmm_model.py         HMM training and comparison (Phase 2)
    lstm_model.py        LSTM architecture and training (Phase 3)
    ensemble.py          Weighted ensemble prediction (Phase 4)
  data/
    raw/                 Downloaded CSVs (gitignored)
    processed/           Cleaned parquet files (gitignored)
  models/                Saved model checkpoints (gitignored)
  notebooks/
```

## Chronological split

| Split | Period | Rows |
|-------|--------|------|
| Train | 2005–2015 | ~3 020 |
| Validation | 2016–2019 | ~1 006 |
| Test | 2020–present | ~1 558 |
