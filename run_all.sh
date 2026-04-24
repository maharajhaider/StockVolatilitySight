#!/bin/bash
set -e
echo "Running Notebook 01..."
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_collection.ipynb

echo "Running Notebook 02..."
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_normality.ipynb

echo "Running Notebook 03..."
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_hmm_regime.ipynb

echo "Running Notebook 04..."
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_lstm_baseline.ipynb

echo "Running Notebook 05..."
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/05_lstm_regime.ipynb

echo "Running Notebook 06..."
venv/bin/python3 src/ensemble.py
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/06_ensemble_eval.ipynb

echo "All complete!"
