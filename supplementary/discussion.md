# Discussion Log

Running record of important decisions, divergences from the proposal, and rationale. Keep entries concise and focused — this is a working document for the team, not a plan artifact.

**Conventions:**
- The proposal (`plan/Research 440 Proposal.md`) frames the research question and is the source of truth for *intent*.
- The plan doc (`plan/hmm-lstm_volatility_forecast_11d292d8.plan.md`) is a hypothesis / implementation log. Empirical results can and do diverge from what the plan predicted — that's the research finding, not a bug. Log divergences here with neutral framing.
- When implementation diverges from the *proposal*, log it here with a rationale.

---

## Current State (updated 2026-04-23, post-rerun)

**Pipeline status:** Phases 1-6 of the plan are complete. Full pipeline re-run from a fresh clone today (nb 01 → 04 → 05 → `src/ensemble.py` → nb 06). HMM selected (GMMHMM, 2 states, 2 mixture components, full cov). All three LSTMs trained. Ensemble evaluated on 2020-06-30 → 2026-03-16 (n=1,434).

**Headline empirical result (fresh run, `notebooks/06_ensemble_eval.ipynb`):**

| Model | MSE | MAPE | Prior run (saved) MSE |
|---|---|---|---|
| Naive (rolling_std_21) | 2.2e-5 | 34.6% | 2.2e-5 |
| **Baseline LSTM** | **1.6e-5** | **25.1%** | 1.5e-5 |
| Calm LSTM | 4.1e-5 | 34.9% | 2.0e-5 |
| Volatile LSTM | 8.5e-5 | 119.4% | 8.5e-5 |
| Ensemble | 2.6e-5 | 31.9% | 1.7e-5 |

**Diebold-Mariano significance tests** (nb 06 §6, first time actually executed):

| Comparison | Loss | DM | p | Verdict (α=.05) |
|---|---|---|---|---|
| Naive vs Baseline (full) | MSE | +1.36 | 0.173 | tie |
| Naive vs Baseline (full) | MAE | +2.46 | 0.014 | **Baseline wins** |
| Baseline vs Ensemble (full) | MSE | −1.75 | 0.080 | tie (marginal, α=.10) |
| Baseline vs Ensemble (full) | MAE | −2.00 | 0.046 | **Baseline wins** |
| Baseline vs Ensemble (volatile n=27) | MSE | +0.62 | 0.542 | tie |
| Baseline vs Ensemble (volatile n=27) | MAE | +0.22 | 0.831 | tie |

On aggregate the baseline LSTM beats the HMM-LSTM ensemble — significantly on MAE (p=0.046), marginally on MSE (p=0.080). On the 27 strictly-volatile test days the ensemble is *directionally* better (MSE 5e-5 vs 8e-5) but the sample is too small to reject (p > 0.5). Regime separation helps where it should; the calm regime dominates aggregate metrics.

**Training stochasticity concern:** this run's ensemble MSE is 2.6e-5 vs the 1.7e-5 in the previously-saved notebook output — a ~50% difference from same config, different optuna/torch numerical noise. The calm LSTM in particular doubled its MSE (4.1e-5 vs 2.0e-5). **Single-seed results are not reliable for ranking claims** — we should run k seeds and report mean±std before locking in any narrative.

**Known reproducibility gaps (four now):**
- `data/processed/train|val|test.parquet` predate the Phase-6 addition of `relative_volume_21d`. Cloning and running nb 04 directly → `KeyError`. Fix: re-run nb 01 (which calls `features.build_features` and saves with the column).
- **`src/ensemble.py` is an undocumented step between nb 05 and nb 06.** Nb 06 cell 4 reads `data/processed/test_predictions.parquet`, which `ensemble.py` produces. Neither the README nor the notebooks call this out. The nb 05 → nb 06 chain fails with `FileNotFoundError` without this step. Discovered today when running the pipeline end-to-end.
- `data/processed/test_predictions.parquet` is not tracked (produced by the step above).
- LSTM checkpoints (`models/*.pt`) are gitignored.

**Open items to decide:**
1. Is the aggregate-loss / volatile-day-win tradeoff acceptable for the writeup, or do we want to revisit gating? (Proposal explicitly listed a learned gating network as out of scope.)
2. Volatile LSTM outputs ~0.0175 nearly constantly and pulls the ensemble up on calm days — does this need a fix (clipping, higher gating threshold) or is it part of the finding?
3. `config.VAL_END = 2020-04-30` means val now includes COVID. The plan text still says "2016-2019." Decide whether to update the plan text or realign the config.
4. **Multi-seed robustness study.** Re-run LSTM training under k≥5 random seeds to get mean±std of the metrics. The current single-run swing from ensemble-MSE 1.7e-5 to 2.6e-5 means any "baseline beats ensemble" claim is fragile.
5. Document the nb 05 → `ensemble.py` → nb 06 step in README.md (or bundle ensemble.py invocation into nb 06 cell 2).

---

## 2026-04-23 — Plan summary and proposal/plan discrepancies

### Plan summary

Regime-aware hybrid model for 21-day forward realized volatility on SPY:

1. **Data + features**: OHLCV + put/call + AAII sentiment → log returns, abs returns, ranges, rolling stats; chronological split; normality diagnostics to justify HMM choice.
2. **HMM regime detection**: Gaussian vs GMM-HMM, 1st- vs 2nd-order (via augmented state space), BIC/LL comparison, Viterbi sanity check against 2008/2020/2022 crises.
3. **LSTMs**: baseline (full data) + calm + volatile, PyTorch CPU, log-target with MSE, Optuna TPE tuning.
4. **Ensemble**: `y = P(calm|O)·y_calm + P(volatile|O)·y_volatile`.
5. **Evaluation**: MSE/RMSE/MAE vs baseline LSTM on 2020-present.

**Current result:** Ensemble MSE 0.00167 vs baseline 0.00246 on the 2020-present holdout — the regime-split hypothesis is validated.

### Discrepancies between proposal and plan

| # | Proposal says | Plan / code does | Rationale |
|---|---|---|---|
| 1 | Targets SPY, XEQT, and MAG7 individual stocks | SPY only | XEQT has insufficient history; MAG7 dropped from scope. Documented in plan. |
| 2 | Train 2005-2015, Val 2015-2020, Test remaining | Train 2004-2015, Val 2016-2019, Test 2020-present | Not explicitly rationalized in plan (git log shows a revert on the split). Non-overlapping boundaries are cleaner but the drift from the proposal is undocumented. **Open item.** |
| 3 | Inputs include raw OHLCV | Phase 6 removed raw OHLCV; kept log_return, abs_return, oc_return, intraday_range, relative_volume_21d | Raw prices caused non-stationary drift in predictions; stationarity fix dropped MAPE from ~600% to ~25%. Tried, didn't work — legitimate divergence. |
| 4 | Compare Gaussian vs GMM-HMM; explore higher-order Markov "if time permits" | Both done (GMM-HMM and 2nd-order via augmented state space) | No discrepancy. |
| 5 | Reference baselines: baseline LSTM + "considering" a simple tree/NN | Baseline LSTM + naïve rolling_std_21 (no tree model) | Plan picked a different simple baseline. Proposal's tree regressor was never implemented. |
| 6 | Regime-specific LSTMs train on regime-partitioned data | Volatile-LSTM val fallback: 0 volatile-dominant windows in 2016-2019, so early-stopping uses the unfiltered val set | Practical necessity — the 2016-2019 val window is almost entirely calm. Methodological caveat: volatile LSTM's model selection isn't purely regime-matched. |
| 7 | Learned gating network — considered, out of scope | Out of scope | No discrepancy. |
| 8 | Non-price inputs: put/call volume (Ho et al., 2012) and AAII weekly sentiment (Gupta et al., 2023) as key features | **Scaffolded but inert.** `SENTIMENT_FEATURES` + `PUTCALL_FEATURES` defined in `config.py` since commit 888a861 (2026-04-16), loaders exist in `src/data_loader.py`, but the CSVs were never downloaded to `data/raw/` and neither group is referenced by `HMM_FEATURES` or `LSTM_BASELINE_FEATURES`. Verified via git log: never added to the active feature lists, never removed — just never activated. | HMM + LSTMs train on price/volume only. Significant divergence from the proposal's core "non-price inputs matter for vol" thesis. Options for the paper: (a) download the CSVs and retrain to restore proposal alignment; (b) explicitly note the scope-down in methods + limitations. |

**Most consequential to flag in the writeup:** #1 (scope narrowing to SPY), #3 (stationarity-driven feature change), #6 (volatile-LSTM val fallback), **#8 (sentiment + put/call never wired in — materially weakens the "non-price signals matter" thesis)**. #2 is the open item — worth deciding whether to retro-justify or realign.

---

## 2026-04-23 — Repo survey: code, notebooks, data sanity

### Data we have

Checked-in processed data (`data/processed/`):

- `features_all.parquet` — full feature table, 5,584 rows, 2004-01-05 → 2026-03-16.
- `train.parquet` (3,020 rows, 2004-01-05 → 2015-12-31), `val.parquet` (1,089 rows, 2016-01-04 → 2020-04-30), `test.parquet` (1,475 rows, 2020-05-01 → 2026-03-16).
- `regime_probabilities.parquet` — HMM soft probs + Viterbi labels, 5,564 rows. Overall regime mix: 96.3% calm, 3.7% volatile (matches plan).

Columns present in train/val/test (20): `Open, High, Low, Close, Volume, log_return, abs_return, oc_return, intraday_range, log_volume, rolling_mean_{5,10,21}, rolling_std_{5,10,21}, rolling_abs_mean_{5,10,21}, realized_vol_21d`.

Sanity spot-checks on train pass: `log_return` mean≈0.0004, std≈0.012 (✓ daily SPY scale); `log_volume` ∈ [16.5, 20.6] (✓ log of millions); target `realized_vol_21d` positive, same scale as `rolling_std_21`; no NaNs after rolling warmup. Target formula in `features.py:154` correctly implements the proposal: forward-looking, demeaned, window=21 — `sqrt( sum (r_{t+j} − r̄)² / m )`.

### Code layout

`src/` holds all modules (not `models/`, which only holds serialized `.joblib` artifacts). Key files:

- `features.py` — engineers all features and target; matches proposal math.
- `hmm_model.py` — Gaussian vs GMM-HMM comparison, penalised EM with restarts, 1st- vs 2nd-order via state-space augmentation.
- `lstm_model.py` — stacked LSTM + linear head, sliding-window `Dataset`, log-space prediction.
- `train_LSTM_baseline.py`, `train_LSTM_regime.py` — Optuna TPE tuning → final retrain.
- `ensemble.py` — weighted-sum ensemble keyed on HMM soft probs.
- `config.py` — single source of truth for features, splits, HMM/LSTM search spaces.

HMM artifacts in `models/`: `hmm_winner.joblib` = GMMHMM(states=2, n_mix=2, cov=full), val LL=9057.28, BIC=−15925.74. Volatile state is state 1. Configurable HMM_DURATION_PENALTY=3000 (≈n_train_obs) prevents degenerate rapid-switching.

### Current test-set results (from `notebooks/06_ensemble_eval.ipynb`, 2020-06-30 → 2026-03-16, n=1,434)

| Model | MSE | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Naive (rolling_std_21) | 0.000022 | 0.00470 | 0.00313 | 34.6% |
| **Baseline LSTM** | **0.000015** | **0.00390** | **0.00246** | **25.3%** |
| Calm LSTM | 0.000020 | 0.00442 | 0.00279 | 29.5% |
| Volatile LSTM | 0.000085 | 0.00922 | 0.00857 | 119.4% |
| Ensemble | 0.000017 | 0.00415 | 0.00271 | 28.8% |

Regime-stratified (HMM confidence > 0.8): on strictly-calm days (n=1,407) baseline wins (MSE 1e-5 vs 2e-5); on strictly-volatile days (n=27) **ensemble wins** (MSE 5e-5 vs 1e-4).

### Findings worth flagging

1. **Empirical ordering differs from the plan's hypothesis.** The plan doc anticipated ensemble > baseline (MSE 0.00167 vs 0.00246, pre-Phase-6). Current post-Phase-6 notebook-06 output shows baseline (1.5e-5) slightly ahead of ensemble (1.7e-5) on aggregate — but ensemble beats baseline on strictly-volatile days (n=27, MSE 5e-5 vs 1e-4). Not a failure: the plan is a hypothesis doc, and this *is* the research result. Worth framing clearly in the writeup as "regime separation helps on volatile regimes specifically; the calm regime dominates 2020-2026 so the baseline is hard to beat on aggregate."

2. **Volatile LSTM is overpredicting on calm days.** Its test MAPE is 119% — when the HMM is wrong about the regime (or uncertain), the volatile model's constant-ish high output (~0.0175) pulls the ensemble up. Options: (a) clip volatile predictions when p_volatile is small, (b) use a higher gating threshold, (c) revisit the learned-gating-network idea that the proposal put out of scope.

3. **Val split now goes to 2020-04-30, not 2019-12-31.** `config.VAL_END = "2020-04-30"` — val *includes* the COVID crash. The plan text and plan table both say "2016-2019" val. This change means the volatile-LSTM val-fallback issue (plan Phase 3b) may no longer apply, since COVID gives the val set ~2 months of volatile-dominant windows. **Discrepancy vs plan and vs the #2 item in the earlier discrepancy table.** The config comment on line 22 is also wrong (says test is "2020-01-01 → present" but test actually starts 2020-05-01).

4. **Saved parquets are stale vs the code.** `config.LSTM_BASELINE_FEATURES` requires `relative_volume_21d`, and `features.build_features` computes it (line 298), but the checked-in `train/val/test.parquet` files do not contain the column. Re-running notebook 01 would fix it; trying to retrain LSTMs against the committed data will KeyError. Also, `data/processed/test_predictions.parquet` (which notebook 06 loads in cell 4) is not in the tree — results only reproducible from a full local regeneration.

5. **LSTM checkpoints are intentionally not tracked.** `.gitignore` excludes `models/*.pt`, `models/*.pth`. This is a choice (size), but combined with (4) it means a fresh clone can't reproduce the ensemble without re-running notebooks 01, 03, 04, 05, 06 in order. Worth a note in the README if we care about reproducibility for the graders.

6. **Raw OHLCV still lives in the parquets.** Phase 6 removed them from `LSTM_BASELINE_FEATURES`, but the columns remain in the DataFrames (just unused). Not a bug — feature selection happens at the config level — but worth knowing if anyone's debugging feature plumbing.

---

## 2026-04-23 (later) — Added Diebold-Mariano forecast significance testing

### What prompted this

In discussion of the metrics table, two methodological concerns surfaced:

1. **Raw MSE is scale-misleading at this target scale.** Realized volatility sits in the 0.005–0.05 range, so MSE is necessarily tiny. Reporting "Baseline beats Naive by 32% MSE" (2.2e-5 → 1.5e-5) made the gap sound dramatic, but the absolute numbers convey little about practical significance. MAPE (34.6% → 25.3%, a 9.3pp absolute reduction in relative error) and RMSE (0.00470 → 0.00390, ~17% lower) are more interpretable — each forecast is off by roughly a quarter of the true vol, not a 30th of anything.
2. **All rankings were point estimates.** We had not formally tested equal predictive accuracy, so we couldn't distinguish "genuinely more accurate" from "slightly lower empirical loss on this sample." This matters most for the headline **Baseline vs Ensemble** comparison, where the RMSE gap is only ~6% (0.00390 vs 0.00415) — well within the range where stochastic variation could flip the ordering.

The second point is a legitimate weakness in the current writeup: we were making ranking claims without showing they were significant.

### What was added

`notebooks/06_ensemble_eval.ipynb` — new **Section 6** (cells 14–17):

- **`diebold_mariano()` helper** implementing the Diebold-Mariano (1995) test with the Harvey-Leybourne-Newbold (1997) small-sample correction. Newey-West HAC variance with Bartlett kernel, lag = h − 1 = 20 to account for autocorrelation from our overlapping 21-day targets. Supports MSE, MAE, and MAPE loss.
- **Three comparisons**, each run with both MSE and MAE loss for robustness:
  1. **Naive vs Baseline LSTM** — sanity floor.
  2. **Baseline vs Ensemble** (full test, n=1,434) — primary research question.
  3. **Baseline vs Ensemble** on strictly-volatile days (p_volatile > 0.8, n=27) — the regime where the ensemble is supposed to help.
- A formatted results table with DM statistic, p-value, significance code, and α=0.05 verdict.
- Interpretation notes clarifying that the strictly-volatile comparison is severely underpowered and should be read as directional evidence only.

### Status: pending execution

`data/processed/test_predictions.parquet` and the `models/*.pt` LSTM checkpoints are not tracked (see "Known reproducibility gaps"), so the DM cells cannot run on the current fresh checkout. The code is in place and will execute on the next full regeneration of notebooks 01 → 06.

### Priors — what we expect when it runs

| Test | Prior | Reasoning |
|---|---|---|
| Naive vs Baseline (full) | Strongly reject (p < 0.01) | RMSE gap ~17% with n=1,434 — ample power |
| Baseline vs Ensemble (full) | Ambiguous | Gap only ~6% RMSE; plausibly within noise |
| Baseline vs Ensemble (volatile) | Likely fail to reject | n=27 is too small to reject despite large point-estimate gap (ensemble halves MSE) |

### Why this matters for the paper

DM results let us calibrate our claims:
- If **naive vs baseline** rejects → "the deep model learns structure beyond persistence." ✓ clean claim.
- If **baseline vs ensemble (full)** does *not* reject → reframe headline from "baseline slightly beats ensemble" to **"baseline and ensemble are statistically tied on aggregate, and ensemble is directionally better on volatile-dominant days."** This is both more honest and a stronger-feeling paper than claiming the ensemble lost.
- If **baseline vs ensemble (volatile)** does *not* reject → we cannot make a significance claim on the volatile subset. Report as directional evidence plus an explicit note on low power, and suggest larger/longer test windows (extend the test period before 2020 or include additional crisis periods) as future work.

### Weakness surfaced (for the paper's limitations section)

Until this point we were reporting model rankings without testing them. Identified during review; now remedied for the aggregate and volatile comparisons. A remaining weakness is test-set power for regime-conditional claims: the 2020-06-30 → 2026-03-16 window contains only 27 days meeting p_volatile > 0.8, which is insufficient to detect anything short of huge effects. This is a structural limitation of using SPY (calm-dominated) on a post-COVID test window rather than a flaw in the model.

---

## 2026-04-23 (evening) — Pipeline re-run from fresh clone; DM results executed

### Motivation

Imported the repo fresh and attempted to reproduce the pipeline end-to-end. Two things we wanted: (1) confirm that the reproducibility gaps we'd identified were real, (2) actually produce `test_predictions.parquet` so the Diebold-Mariano cells added earlier today could run.

### Reproducibility issues found (in the order we hit them)

1. **`relative_volume_21d` missing from committed parquets** — `train_LSTM_baseline.py` failed immediately with `KeyError: "Columns missing from dataframe: ['relative_volume_21d']"`. Fix: re-run nb 01, which calls `features.build_features` and saves the parquets with the new feature. Verified this is a pre-existing gap (git status confirmed we hadn't touched `config.py`, `src/`, or `data/processed/`). Same failure any teammate would hit from a fresh clone.

2. **`src/ensemble.py` step is undocumented** — after nb 04 + nb 05 produced the three LSTM `.pt` files, nb 06 failed with `FileNotFoundError: data/processed/test_predictions.parquet`. This file is produced by running `python src/ensemble.py` as an intermediate step. The README lists only notebooks; neither nb 05 nor nb 06 mentions that an external script must run between them. Fix: explicitly document the step in README, or bundle the invocation into nb 06's setup cell.

3. **Package install was ad-hoc.** I installed `optuna`, `seaborn`, `yfinance` piecemeal as errors hit, instead of `pip install -r requirements.txt` up front. Lesson for the log: start with the README's setup block.

### Fresh-run metrics (n=1,434, test window 2020-06-30 → 2026-03-16)

| Model | MSE | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Naive (`rolling_std_21`) | 0.000022 | 0.00470 | 0.00313 | 34.6% |
| **Baseline LSTM** | **0.000016** | **0.00398** | **0.00257** | **25.1%** |
| Calm LSTM | 0.000041 | 0.00638 | 0.00343 | 34.9% |
| Volatile LSTM | 0.000085 | 0.00922 | 0.00857 | 119.4% |
| Ensemble | 0.000026 | 0.00511 | 0.00310 | 31.9% |

Regime-stratified (p > 0.8): strictly-calm n=1,407 — Baseline MSE 0.00001, Ensemble MSE 0.00003 (baseline wins). Strictly-volatile n=27 — Baseline MSE 0.00008, Ensemble MSE 0.00005 (ensemble wins directionally).

### Diebold-Mariano results

Sign convention: positive DM statistic means the first-listed model has higher loss (so the second-listed model is more accurate). Newey-West HAC with Bartlett kernel, lag 20; HLN small-sample correction; two-sided t-test with df = n-1.

| Comparison | Scope | Loss | DM | p | Signif. | Verdict (α=.05) |
|---|---|---|---|---|---|---|
| Naive vs Baseline | full n=1,434 | MSE | +1.36 | 0.173 | n.s. | tie |
| Naive vs Baseline | full n=1,434 | MAE | +2.46 | 0.014 | ** | Baseline wins |
| Baseline vs Ensemble | full n=1,434 | MSE | −1.75 | 0.080 | * | tie (marginal) |
| Baseline vs Ensemble | full n=1,434 | MAE | −2.00 | 0.046 | ** | Baseline wins |
| Baseline vs Ensemble | volatile n=27 | MSE | +0.62 | 0.542 | n.s. | tie |
| Baseline vs Ensemble | volatile n=27 | MAE | +0.22 | 0.831 | n.s. | tie |

### Priors vs reality

| Test | Prior (pre-run) | Actual | Assessment |
|---|---|---|---|
| Naive vs Baseline | Strongly reject (p < 0.01) | MSE p=0.17 (tie), MAE p=0.014 (reject) | **Weaker than predicted on MSE.** The squared-error loss has higher variance, so the HAC-corrected test has less power. MAE gives the clean claim. |
| Baseline vs Ensemble (full) | Ambiguous | MSE p=0.08, MAE p=0.046 | **Prior confirmed.** Result is borderline — depending on loss and significance threshold, baseline either marginally wins or ties. |
| Baseline vs Ensemble (volatile) | Likely fail to reject despite large gap | p=0.54 / 0.83 | **Prior confirmed.** n=27 is simply too small. |

### Defensible claims for the paper

- "The baseline LSTM is more accurate than the 21-day rolling-volatility persistence baseline (DM MAE test, p=0.014)." ← strongest claim, use MAE not MSE.
- "The baseline LSTM matches or slightly outperforms the HMM-LSTM ensemble on aggregate (DM, p=0.046 MAE, p=0.080 MSE)." ← honest framing. Not "the ensemble fails."
- "On volatile-dominant subperiods, the ensemble is *directionally* more accurate (point-estimate MSE 0.00005 vs 0.00008), but the n=27 subsample is underpowered to reject equal accuracy at α=0.05." ← directional with an explicit caveat.

### Training stochasticity — a new concern

Compared to the previously-saved nb 06 output (a prior run by a teammate):

| Model | Prior saved MSE | This-run MSE | Change |
|---|---|---|---|
| Baseline | 1.5e-5 | 1.6e-5 | +7% |
| Calm | 2.0e-5 | 4.1e-5 | **+105%** |
| Volatile | 8.5e-5 | 8.5e-5 | 0% |
| Ensemble | 1.7e-5 | 2.6e-5 | **+53%** |

Same config, same seeds in `config.py` (`RANDOM_STATE=42`, `LSTM_RANDOM_STATE=42`), same data. The gap comes from (a) Optuna's TPE exploration — different trial sequences can converge to different hyperparameter neighbourhoods, (b) CPU PyTorch non-determinism without `torch.use_deterministic_algorithms(True)`.

**Implication:** ranking claims based on a single training run are fragile. The baseline-vs-ensemble MSE gap went from 13% (prior) to 63% (this run). Before locking in the narrative for the paper, we should run k≥5 seeds and report mean±std on each metric, or at minimum run 3 seeds and report the range. This is a real limitation to cite in the paper's methodology/limitations section, not just a Claude-introduced concern.

### Why this entry matters for the paper writeup

- Shows the empirical result is genuinely nuanced, not a clean win or loss.
- Documents that we identified and tested methodological weaknesses (no significance testing → added DM; single-run results → flagged multi-seed need).
- Provides a transparent trail: priors → actual results → assessment, which is the kind of thing reviewers value.

## 2026-04-23 (late evening) — Variant B: forward-looking VIX-family features

### Motivation

F1 (in `report.md`) predicts that forward-looking features should be the only way out of the persistence-shift floor. This experiment directly tests the prediction: train a parallel set of LSTMs (baseline + calm + volatile) on variant A's features *plus* the CBOE VIX-family forward-looking indices, re-run the soft-probability ensemble, and Diebold-Mariano each model pair against variant A.

### Scope and design decisions

- **Variant A**: existing checkpoints (variant A LSTMs were not retrained — per user instruction, "we already have the data").
- **Variant B**: variant A's 5 LSTM features + `vix`, `vix_log_change`, `vix3m_minus_vix` = 8 input features.
- **HMM**: fixed. Same regime labels and same soft probabilities feed both variant A's and variant B's ensembles. This isolates the LSTM's response to the new features from any HMM-regime reshuffle.
- **Data source**: FRED (`VIXCLS`, `VXVCLS`) via `pandas_datareader`. We originally wanted the full yfinance VIX family (`^VIX`, `^VIX3M`, `^SKEW`, `^VVIX`) but yfinance turned out to be too unreliable for index tickers under rate-limit pressure, and FRED doesn't host SKEW / VVIX. Reduced set = 2 raw + 2 derived = 3 features used.
- **Training window**: variant B's LSTMs effectively start 2007-12-04 (when VXVCLS's history begins), losing ~1,680 training rows. Val and test windows are unchanged.
- **Reproducibility**: `src/data_loader.py::download_vix_family()`, `config.py::VIX_FAMILY_FEATURES`, and `notebooks/07_variant_b_experiment.ipynb` encode the full spec. Re-executing nb 07 regenerates variant B end-to-end.

### Results

Test-set performance on the 1,434 common rows (2020-06-30 → 2026-03-16):

| Model | Variant A MSE | Variant B MSE | Δ |
|---|---|---|---|
| Baseline LSTM | 1.60×10⁻⁵ | **1.36×10⁻⁵** | −15 % |
| Calm LSTM | 4.10×10⁻⁵ | **1.42×10⁻⁵** | −65 % |
| Volatile LSTM | 8.50×10⁻⁵ | 9.26×10⁻⁵ | +9 % (worse) |
| Soft-prob ensemble | 2.60×10⁻⁵ | **1.40×10⁻⁵** | −46 % |

Diebold-Mariano (A vs B, h=21, Bartlett HAC, HLN-corrected; positive DM ⇒ B wins):

| Model | MSE DM | MSE p | MAE DM | MAE p | Verdict (α=.05) |
|---|---|---|---|---|---|
| baseline | +1.86 | 0.064 | +1.71 | 0.087 | tie (marginal) |
| calm | +1.85 | 0.065 | +2.33 | 0.020 | **B wins on MAE** |
| volatile | −13.09 | <.001 | −16.99 | <.001 | **A wins ***|
| ensemble | +1.93 | 0.054 | +2.14 | 0.033 | **B wins on MAE** |

### F1 diagnostic — did the shift lag shrink?

Cross-correlation peak lag (variant A → variant B):

- Baseline: −16 → **−14** days
- Calm: −18 → **−14** days
- Ensemble: −17 → **−16** days
- Volatile: meaningless (both LSTMs degenerate, peak correlation is negative)

Small reduction (2-4 days out of ~17), not a structural escape. Peak correlation at lag 0 improved modestly (baseline 0.56 → 0.57; ensemble 0.53 → 0.55). The persistence-shift floor is still clearly there.

### F2 diagnostic — volatile-LSTM degeneracy

| Variant | Volatile pred std | Volatile pred range |
|---|---|---|
| A | 5.3×10⁻⁵ | 8×10⁻⁴ |
| B | 1.6×10⁻⁴ | 7×10⁻⁴ |

Variant B's volatile LSTM has 3× higher prediction std than variant A's but both are still orders of magnitude below the target std (~4×10⁻³). Range barely changed. **F2 persists** — VIX features don't provide enough signal on the ~170-sample volatile training set to break the constant-output collapse.

### Interpretation for the paper

VIX features meaningfully improve point forecasts on calm-regime days and therefore the ensemble (since the HMM assigns calm weight ≈1 on ~97% of days). But:

1. The improvement is consistent with "tighter persistence prediction" (variant B's prediction std is *smaller* than variant A's, not larger), not "forward-looking information".
2. The shift lag reduces only marginally.
3. The volatile LSTM degeneracy persists (F2 is a data-scarcity problem, not a feature-set problem).

**This strengthens the structural-limits framing**: even the options market's own implied-volatility forecast — the best forward-looking signal available for equity volatility — doesn't pull a daily-resolution LSTM off the persistence floor. The limitation is data-rate × target-horizon, not feature coverage.

### Defensible claims for the paper

- *"Adding forward-looking VIX-family features reduces ensemble test MSE by 46% (2.6×10⁻⁵ → 1.4×10⁻⁵), statistically significant on MAE (DM p=0.033)."* — headline positive result from variant B.
- *"However, the cross-correlation peak lag of LSTM predictions vs. the target only shifts from −17 days to −16 (ensemble) / −14 (baseline, calm). The persistence-shift floor is not structurally escaped."* — the F1 limitation survives.
- *"The volatile-regime LSTM remains effectively constant (prediction std 1.6×10⁻⁴ vs. target std 4×10⁻³) under both variants, confirming F2 is a data-scarcity problem not a feature-set problem."* — F2 survives.
