# Report — Don't-Miss Checklist

Working notes mapping our project work onto the CPSC 440/550 report outline (from `instructions/project.pdf`). Focus is on **Section 3 (Description & Justification)** and **Section 4 (Experiments & Analysis)**. HMM/data coverage first; LSTM and ensemble to be added later.

**Paper length: up to 6 pages of main content.** Anything extra goes to supplementary material (no page limit). Figures > tables with many numbers; well-designed color-coded tables are OK.

---

## ⚑ Headline findings — what the paper should lead with

Two diagnostic findings reframe the entire research narrative. These are the most important things not to bury.

### F1 — All LSTMs are persistence predictors; predictions are shifted ~17 days late

Cross-correlation of each predictor against the target peaks at **negative lags** of 16-21 days:

| Model | Zero-lag corr | Best lag | Best-lag corr |
|---|---|---|---|
| Naive (rolling_std_21) | +0.46 | **−21 days** | **+1.00** (mathematically identical) |
| Baseline LSTM | +0.56 | −16 days | +0.83 |
| Calm LSTM | +0.46 | −18 days | +0.82 |
| Ensemble | +0.53 | −17 days | +0.70 |

Negative lag = the prediction is *shifted later than* the target = the model is *reactive*, not *predictive*. The naive baseline is literally the target time-shifted back 21 days (rolling_std_21 at time t and realized_vol_21d at time t−21 are computed from the same 21 returns). Our LSTMs achieve 0.82-0.83 correlation at lag −16 — they're slightly-worse persistence predictors.

**Implication:** on a 21-day-forward target with price/volume features only, there's no forward-looking signal to extract — the gradient-trained optimum is "output something close to recent vol." The deep model captures ~70% of the variance the naive captures.

**Why this is structural, not architectural.** Future returns are not predictable from past returns (anything close to Efficient Markets Hypothesis). Future-magnitude (vol) is modestly predictable from past-magnitude, but only via persistence. No LSTM architecture on price/volume inputs can escape this — the shift is a *data* property, not a model property.

**Tested directly via three post-sentiment-merge variants (see §4.5 and `notebooks/07_variant_comparison.ipynb`):**

- **Variant A** — stationary + sentiment (7 feats), regime-split ensemble
- **Variant H** — variant A + `p_volatile` as an LSTM input feature (8 feats), single baseline LSTM (no ensemble gating)
- **Variant B** — variant A + VIX family (10 feats: `vix`, `vix_log_change`, `vix3m_minus_vix`), regime-split ensemble

All three are **statistically tied on MSE and MAE at h = 21** (pairwise DM p ≥ 0.27). Peak cross-correlation lags: **−16 (A, H)** and **−17 (B)** — adding VIX pulls the lag *backward* by one day, not forward. Handing the LSTM the HMM regime probability directly (H) reproduces the same −16-day shift that the gated ensemble (A) has. The persistence-shift floor survives every augmentation tested, including the options market's own implied-vol forecast.

**What would actually escape F1:** the shift is data-rate × horizon. Higher-frequency bars (5-min) or a shorter target horizon (1-day) would change the problem, not just the features. See Future Work.

### F2 — The Volatile LSTM is structurally degenerate (essentially constant)

| Model | Pred std | Pred range |
|---|---|---|
| Target | 0.0044 | — |
| Baseline | 0.0039 | 0.007 → 0.031 |
| **Volatile** | **0.0001** | **0.0175 → 0.0183** |

The Volatile LSTM's predictions vary by less than 0.8 basis points across the whole test set — it outputs ~0.0179 regardless of input (the mean of its 181 training targets). With a 17k-parameter LSTM on 181 windows, Optuna found "output the mean" as the best solution. When `p_volatile > 0` in the ensemble this constant pulls predictions up toward 0.018, way above typical calm-day vol (~0.008), actively contaminating the ensemble.

**F2 has two distinct failure modes, both driven by data scarcity.** The volatile-LSTM Optuna training converges to one of two pathologies depending on the seed / trial path:

- **Mode 1 — near-constant output** (variant A post-sentiment run, and every earlier run we've done): prediction std ~10⁻⁴, range <10⁻³. Outputs the mean of its ~180 training targets.
- **Mode 2 — wild non-stationary output** (variant B post-sentiment run): prediction std 0.65, max 2.71 (implausible 271 % daily vol). Model fails to converge at all in the volatile subspace.

Both modes are symptoms of the same underlying problem: with only ~180 training windows dominated by a single 2008-GFC-shaped event (F4), Optuna's hyperparameter search can land in either "output the mean" or "output noise" local optima with comparable probability, and both look bad on test. The soft-probability ensemble gating partially masks the damage in both cases because `p_volatile` is small on ~97 % of days — but the volatile LSTM is contributing no useful signal in either variant.

**The seed-dependence itself is the finding.** If F2 were a feature-set problem, richer features would consistently help; if it were a hyperparameter problem, Optuna would reliably find a stable optimum. Instead the failure mode flips between runs. That's the signature of a data-scarcity problem (F4) — which is why none of variants A / H / B escape F2.

### Combined framing for the paper

The regime-switching hypothesis (HMM-LSTM > plain LSTM) requires the regime signal to add information *beyond* recent vol. On our data that additional information simply isn't there — the regime label at time t is almost fully redundant with the recent-magnitude features the baseline LSTM already sees. **Our ensemble is "two reactive predictors gated by a reactive detector"; it's structurally incapable of beating persistence on a 21-day-forward task with only price/volume features.**

This is a legitimate, well-evidenced negative result — and the *honest finding* of the project. Frame it as such rather than trying to salvage a positive claim.

### F3 — A learned linear gate cannot rescue the ensemble either

To test whether the HMM-probability weighting was the weak link (vs the architecture itself), we replaced the soft-probability ensemble with a **learned Ridge stacking meta-learner** taking all three LSTM predictions + both HMM probabilities as inputs and fit to predict the target. (`src/gate_stacking.py`, Ridge α swept on val.)

**Result:** gated test MSE = 1.87e-5 vs baseline 1.59e-5. **The learned gate is statistically tied with baseline** (DM MAE p=0.57) and strictly worse on point estimate. Learned coefficients assign near-zero weight to both the Volatile LSTM (−0.002) and the HMM probabilities (±0.004); Ridge explicitly learned to ignore them.

**Why this is the definitive verdict on the regime-split approach**: two independent structural problems make *any* gating mechanism degenerate on this data:

1. **The gating signal barely varies** — `p_volatile > 0.5` on only 3.7% of days. For 97% of days the allocation is fixed at (1.0, 0.0); there's no gating *decision* to make.
2. **The "volatile expert" is a constant** (F2) — when the gate *would* want to activate it, that LSTM outputs ~0.0179 regardless of input. It contributes no information, only a biased offset.

A nonlinear MLP gate (originally planned as Phase 2) would face exactly the same two constraints — there's no nonlinear structure to discover because the inputs that gating would depend on have been empirically zeroed out. **Phase 2 was skipped on these grounds.** Ridge's direct confirmation ("ignore these inputs — they add nothing") is cleaner than any MLP result could be.

### F4 — Structural data scarcity: rare regimes × daily frequency × 21-day horizon

The Volatile LSTM's degeneracy (F2) — and by extension the gate failure (F3) — isn't a training-hyperparameter problem. It's a structural limit of the data. Three independent counts of "effective volatile training events" all land in the single digits:

- **Volatile days in training**: 134 / 3,000 (4.5%), almost entirely from 2008 GFC (107 contiguous days). COVID falls in val, not train, so the volatile LSTM trains on essentially **one crisis**.
- **Majority-volatile sliding windows (n=181)**: consistent with the day count, but windows share 41/42 days → they are not independent.
- **Independent volatile events**: 134 volatile days / 42-day window ≈ **4-5 independent volatile episodes** in training.
- **At the target's natural horizon** (21-day-forward), 3,000 daily sliding windows collapse to ~143 independent monthly observations — of which ~7-14 are volatile-dominant.

For a 17k-parameter LSTM, you want *orders of magnitude* more diverse examples than single-digit independent volatile events. The model memorizes "GFC-like days have vol ≈ 0.018" and outputs that constant ever after. No hyperparameter tuning, no regularization, no gating mechanism can fix this — **it's a data availability problem, not a model problem.**

Three axes along which the problem could be made tractable (none of which we could do within scope):

| Approach | Gain | Cost |
|---|---|---|
| Higher-frequency data (5-min bars) | 78× more per day; volatile *windows* become plentiful | Different problem — microstructure noise dominates; target semantics change |
| Extend history to pre-2004 | Picks up dot-com 2001, LTCM 1998 | Base rate ~4% is structural, not artefactual — 20 years still gives only ~300 volatile days |
| Transfer learning across assets (SPY, QQQ, IWM, international) | Orders of magnitude more data in principle | US-equity regimes are heavily correlated → "more assets" ≈ mild augmentation, not true multiplication |

**The sharper paper framing this enables:** instead of "our regime-split didn't beat baseline," the thesis becomes *"The regime-switching framework with per-regime deep models faces a fundamental data-scarcity problem on daily-frequency equity data. Volatile regimes are rare (~4% base rate), concentrated in individual crises (GFC dominates our training), and — given a monthly-scale prediction horizon — the effective number of independent volatile training events is single-digit. This is a structural limit on the approach, not a flaw in our implementation. We identify three axes along which the problem could be made tractable."*

---

## Section 3 — Description and Justification

### 3.1 Dataset and Feature Selection

- [ ] **What data**: SPY daily OHLCV via yfinance (`auto_adjust=True`), 2004–2026 (~5,584 trading days after dropping rolling-warmup and forward-target NaNs).
- [ ] **Why SPY, not the proposal's broader scope** (XEQT + MAG7 were in the original plan): XEQT has insufficient history; MAG7 individually dropped. **This is a divergence from the proposal — call it out.**
- [ ] **Adjusted vs raw close**: why adjusted (splits/dividends) matters for vol calculations. Brief note.
- [ ] **Non-price data scaffolded but not activated** (sentiment + put/call): the loaders and config groups exist (per Ho et al. 2012 and Gupta et al. 2023 in the proposal) but the CSVs were never downloaded and the groups aren't wired into HMM_FEATURES or LSTM_BASELINE_FEATURES. **Major divergence from the proposal's core "non-price signals matter" thesis. Acknowledge in limitations.**
- [ ] **Chronological split**: train 2004-01 → 2015-12 (3,020 days), val 2016-01 → 2020-04 (1,089 days, *includes COVID*), test 2020-05 → 2026-03 (1,475 days). Emphasize no shuffling (temporal leakage avoided).
- [ ] **Val boundary note**: original proposal said val=2015–2020. Current code has `VAL_END=2020-04-30` so COVID sits in val. This actually helps — it gives the volatile LSTM some volatile-dominant val windows for early stopping (original plan had 0).
- [ ] **Target variable**: 21-day forward realized volatility, σ_t = sqrt((1/m) · Σ(r_{t+j} − r̄)²), m=21. Adapted from Andersen & Bollerslev (1998) for daily (non-high-frequency) data — hence the demeaned form. **Note 1/m vs 1/(m-1) choice** (our code uses 1/m; proposal garbled formula is ambiguous; state explicitly which we use).
- [ ] **Feature list** (post-pruning): log_return, abs_return, oc_return, intraday_range, log_volume, relative_volume_21d, rolling_mean/std_{5,10,21}. Drop rolling_abs_mean_{5,10,21} on correlation > 0.95 with rolling_std_{5,10,21}.
- [ ] **Normality diagnostics motivation**: three tests (Shapiro-Wilk, Jarque-Bera, D'Agostino-Pearson) triangulate. All 11 HMM features reject normality at p < 10⁻³¹. Key numbers: log_return skew = −0.31, excess kurtosis = +15. **Heavy tails → GMMHMM preferred over plain Gaussian.**
- [ ] **Volatility clustering evidence**: Ljung-Box on r² at lags 5/10/21 all p = 0.0. ACF plot of r². This is the statistical mandate for the regime-switching model — without it, the whole architecture is unmotivated.
- [ ] **Realized vol distribution**: raw target is heavily right-skewed (skew +3.52, kurt +17.4); log-transform brings it much closer to normal (skew +0.75, kurt +1.07). Justifies `LOG_TRANSFORM_TARGET = True` for LSTM training.
- [ ] **Feature scaling rationale**: raw feature ranges differ ~18,000:1 (log_volume ≈ 18 vs log_return ≈ 0.001). Breaks k-means init and full-covariance EM conditioning. Use `StandardScaler` fit on train only.
- [ ] **Why not RobustScaler**: evaluated as theoretically attractive (IQR < std preserves crisis signal) but empirically caused degenerate EM convergence. Worth mentioning as a documented negative-result choice.
- [ ] **Stationarity decision for LSTM inputs (Phase 6)**: original inputs included raw OHLCV + log_volume; predictions drifted upward massively (MAPE ~600%). Swap to stationary inputs (relative_volume_21d replacing log_volume; drop raw OHLCV) → MAPE ~25%. **Major methodological finding — either in 3.1 or 3.3 depending on narrative flow.**
- [ ] **Regime-conditional return distributions**: pseudo-regime split on median realized vol shows Levene's test p = 2.8 × 10⁻⁹³ (variances massively different), Welch's t-test p = 9 × 10⁻⁸. Supports regime-aware modelling independent of the HMM.

### 3.2 HMM Component

- [ ] **Model class and why**: GMMHMM (mixture emissions per state), 2 hidden states, full covariance. Selected via val BIC.
- [ ] **Cite Rabiner (1989)** for EM/Viterbi/forward-backward foundations and **Zhang et al. (2019)** for GMMHMM on financial data.
- [ ] **Hyperparameters**: n_states=2, n_mix=2, n_iter=200, n_restarts=50, duration_penalty λ=3,000.
- [ ] **Training protocol**:
  - 50 random restarts per candidate model. Restart 0 uses smart init; restarts 1–49 use hmmlearn default random k-means.
  - Max 200 EM iterations per fit.
  - Standardization fit on train only.
- [ ] **Novel contribution 1 — volatility-quantile smart init**: seeds EM from labels based on abs_return quantiles instead of feature-space k-means. Addresses the problem that k-means clusters by absolute feature level (especially log_volume) and produces already-degenerate initial transition matrices. **This is a meaningful methodological contribution worth highlighting.**
- [ ] **Novel contribution 2 — penalised restart selection**: LL − λ · Σ(1 − p_stay_s) with λ ≈ n_train. Without this, EM can converge to rapid-switching local optima with higher raw LL than the economically sensible split. **Cite Nystrup, Hansen & Madsen (2015–2019).**
- [ ] **Selection criterion**: validation BIC (not training BIC, not raw LL). Lower is better. Penalizes parameter count.
- [ ] **Post-hoc regime labeling**: state with higher mean abs_return → "volatile." Not learned, just assigned after training so the two states have consistent names.
- [ ] **What we learned about the data from the HMM**: calm avg duration ≈ 233 days (11 months); volatile avg duration ≈ 12 days. Regime split 96.3% calm / 3.7% volatile over full timeline. This is an important characterization of SPY regimes itself, independent of downstream forecasting.
- [ ] **Sanity check — economic alignment**: volatile state correctly captures 2008 GFC (107 days, onset the trading day after Lehman), 2020 COVID (47 days, P(vol) ≈ 1.0 at peak), weakly flags 2022 Fed hikes (2 days only). Zero false positives in 2016–2019 benign window. Report this as a figure (SPY close + shaded volatile periods + P(vol) panel).

### 3.3 LSTM Component

- [ ] **Architecture**: stacked `nn.LSTM` + linear head producing a scalar, taking the final timestep as the regression summary. Three separately trained networks:
  - **Baseline LSTM** — trained on all 3,020 training windows.
  - **Calm LSTM** — trained on ~2,819 windows whose majority Viterbi label is calm.
  - **Volatile LSTM** — trained on only 181 windows labeled volatile by majority Viterbi vote. **This is an order of magnitude too little data** — call out the scarcity explicitly.
- [ ] **Target transform**: `log(realized_vol_21d)` trained with MSE loss; `exp()` back for reporting. Log-transform justified by the 3.52 → 0.75 skew reduction from nb 02.
- [ ] **Feature selection** (post-Phase-6): log_return, abs_return, oc_return, intraday_range, relative_volume_21d. **Why these 5 and not the HMM's 11**: LSTM gets context via its sliding window (seq_len=21 or 42), so rolling stats are redundant; `relative_volume_21d` replaces `log_volume` for stationarity (Phase 6 fix) — worth one sentence.
- [ ] **Scaling**: `StandardScaler` fit on full training split (not regime-filtered) for all three models. Ensures calm and volatile LSTMs see the same feature scale.
- [ ] **Hyperparameter tuning**: Optuna TPE with 20 trials × 40 tune-epochs + 100 final-epochs. Search space: hidden_size ∈ {32, 64, 128}, n_layers ∈ {1, 2, 3}, dropout ∈ [0, 0.5], lr ∈ [1e-4, 1e-2] log-uniform, batch_size ∈ {32, 64, 128}, seq_len ∈ {21, 42}.
- [ ] **Regime-window labeling**: each sliding window labeled by the *majority* Viterbi state across its `seq_len` timesteps. Windows straddling regime boundaries get assigned by majority vote — noisy labels exactly at the transition periods we care most about.
- [ ] **Val loader fallback**: if a regime has <1 val window at the chosen seq_len, fall back to the full unfiltered val loader for early stopping. Originally triggered by the 2016-2019 val window (0 volatile days); now mostly unused since VAL_END was extended into COVID.

### 3.4 Ensemble

- [ ] **Weighting formula**: `ŷ_t = p_calm(t) · ŷ_calm(t) + p_volatile(t) · ŷ_volatile(t)`, where `p_{·}(t)` are the HMM forward-backward soft probabilities at time t.
- [ ] **Contrast with Jiang et al. (2023) HMM-ALSTM**: they add HMM states as an extra feature to a single LSTM. We *partition* the data and train separate LSTMs per regime. This is the proposal's novelty over that line of work.
- [ ] **Naive baseline for comparison**: `rolling_std_21(t)` as a pure-persistence reference. **Acknowledge upfront** that this is mathematically the target shifted back 21 days and therefore correlates perfectly at lag −21 — it's a *hard* baseline to beat.
- [ ] **What went wrong (see Headline Findings F1 + F2)**: the Volatile LSTM is degenerate, and all LSTMs converge to ~persistence behavior. The ensemble inherits both problems.

---

## Section 4 — Experiments and Analysis

**Structure principle (from instructions)**: each experiment tests *one thing*, controls other factors. Prefer figures over tables-of-numbers. Color-coded tables OK when numbers are load-bearing.

### 4.1 HMM Experiments

- [ ] **Experiment A — Gaussian vs Gaussian-mixture emissions**
  - Question: does adding mixture components per state improve held-out fit?
  - Setup: n_states=2 fixed, full cov fixed, sweep n_mix ∈ {1, 2, 3}. Same data, same restart protocol, same penalized-LL selection.
  - Result: GMMHMM(n_mix=2) wins val BIC by 1,028 over n_mix=3, and by 34,392 over GaussianHMM(n_mix=1). **n_mix=3 over-parameterizes** (adds 156 params for only +32 val LL).
  - Visual: bar chart of val LL, val BIC, val AIC across the three variants.
  - **Key finding**: mixture helps (mix=2 > Gaussian alone) but more mixture doesn't keep helping (mix=3 worse than mix=2 by BIC). Mixture components end up capturing sub-patterns within each regime (e.g., calm-up vs calm-down, mild-vol vs extreme-vol).

- [ ] **Experiment B — 1st-order vs 2nd-order Markov structure**
  - Question: does higher-order state memory improve fit?
  - Two strategies tested:
    - **Strategy A (state-space augmentation)**: GaussianHMM with 4 composite states representing (s_{t-1}, s_t).
    - **Strategy B (feature-space augmentation)**: 2-state GaussianHMM on 22-dim observations [o_t, o_{t-1}].
  - Decision threshold: retain 1st-order unless 2nd-order improves val BIC by ≥ 10.
  - Result: Strategy A val BIC is +23,862 *worse*; Strategy B is +38,961 *worse*. 1st-order retained.
  - **Key finding**: equity regimes are well-described by a simple 2-state 1st-order HMM; adding temporal memory just burns parameters.
  - **Cite Lee (2017)** for the state-augmentation approach.

- [ ] **Experiment C — Degenerate-EM defense: smart init + penalized restart selection**
  - Question: does our two-layer defense produce economically sensible regimes vs raw EM?
  - Evidence: without smart init or λ-penalty, random restarts often converge to rapid-switching solutions (see "Model is not converging" warnings in training logs — you can point to these).
  - The healthy solution has volatile avg duration ≈ 12 days (plausible for crisis periods) vs degenerate ≈ 2 days (not economically sensible).
  - **Visual**: side-by-side comparison of learned transition matrices — degenerate vs our winner.

- [ ] **Experiment D — RobustScaler vs StandardScaler**
  - Question: does a scaler less sensitive to outliers help EM stability?
  - Result: RobustScaler inflated crisis-period feature magnitudes (because IQR < std), pushed covariance matrices toward near-singularity, and caused EM to converge to degenerate absorbing-state solutions even with many restarts.
  - **Key finding**: StandardScaler is the empirically right choice here, even though RobustScaler was theoretically attractive. Good negative-result to include — shows we thought about it.

- [ ] **Experiment E — Teammate's negative results on reduced features and 3-state HMM** *(if you can get the numbers from Changjiang)*
  - 3-state HMM (crisis/moderate/calm): only 8 training days assigned to "calm" — degenerate.
  - Reduced feature set: only 2 training days assigned to "calm" — degenerate.
  - **Key finding**: both configurations collapsed because (a) SPY is heavily calm-dominated (96.3% in 2-state model), leaving tiny budget for additional regimes to be sensible, and (b) removing features strips the signal needed to separate the already-tiny volatile class.
  - **This is valuable negative-result evidence. Defends "why 2 states + full features" against reviewer pushback.**

### 4.2 Data / EDA Experiments

- [ ] **Experiment F — Triangulated normality testing**
  - Three complementary tests (Shapiro-Wilk, Jarque-Bera, D'Agostino-Pearson) × 11 features. All reject normality. Strong multi-test evidence → justifies mixture emissions.

- [ ] **Experiment G — Ljung-Box test on r² (volatility clustering)**
  - Tested at lags 5, 10, 21. All p = 0.0. Confirms volatility clustering — the statistical premise of the whole project.

- [ ] **Experiment H — Regime-conditional variance test (pseudo-regime via median vol split)**
  - Levene's test on variance equality between halves: p = 2.8 × 10⁻⁹³. Welch's t-test on mean equality: p = 9 × 10⁻⁸. Regimes have structurally different distributions — supports separating them.

### 4.3 LSTM Experiments

- [ ] **Experiment I — Persistence-predictor diagnostic (lag analysis)** *(headline finding F1)*
  - Question: are our LSTM predictions tracking the target in time, or are they just echoing recent history?
  - Setup: compute cross-correlation of each predictor (naive, baseline, calm, volatile, ensemble) against the target at lags ∈ [−30, +30] days. Report the zero-lag correlation and the lag that maximizes correlation.
  - Result: best lag is **negative** (−16 to −21 days) for every predictor. Naive correlates 1.00 at lag −21 (mathematical identity — same 21 returns, different label position). Baseline LSTM correlates 0.83 at lag −16. **LSTM = slightly-smoothed persistence predictor with ~17-day shift.**
  - Visual: table of (model, zero-lag corr, best-lag, best-lag corr). Optionally overlay three time-series panels — target, baseline, baseline-shifted-by-16-days — showing the shift collapses at the right lag.
  - **Key finding**: on 21-day-forward vol with price/volume features, there's no forward-looking signal to extract. Gradient descent's unique optimum is "ŷ(t) ≈ f(recent_vol_at_t)".

- [ ] **Experiment J — Volatile LSTM collapse to constant output** *(headline finding F2)*
  - Question: does the Volatile LSTM actually learn regime-specific dynamics, or does it degenerate?
  - Setup: measure the std and range of each LSTM's predictions across the test set, compared to the target.
  - Result: Volatile LSTM pred std = 0.0001 (range 0.0175 → 0.0183 = 0.8 basis points). Target std = 0.0044. **The Volatile LSTM varies ~40× less than the target**; it's outputting essentially a constant. With only 181 training windows against ~17k parameters, Optuna converged to "output the training mean."
  - Visual: side-by-side histogram of each model's predictions vs the target; volatile LSTM is a spike.
  - **Key finding**: the regime-split architecture is bottlenecked by the volatile regime's data scarcity. Separating architectures doesn't help if one side can't be trained.
  - **Methodological consequence**: when `p_volatile > 0` in the ensemble, this constant drags predictions upward — contaminating the ensemble on calm days where `p_volatile` is small but non-zero.

- [ ] **Experiment K — Naive persistence as a hard baseline (sanity)**
  - Question: how strong is pure persistence as a baseline?
  - Setup: compare the naive `rolling_std_21` prediction to the target directly.
  - Result: Naive zero-lag correlation is +0.46, but at lag −21 it correlates **perfectly** (1.00). This is a mathematical identity — same 21 returns, different label position.
  - **Implication for how to report results**: the naive is not "a weak baseline our deep model crushes" — it's the target shifted 21 days earlier. Any deep model that does better than naive must be extracting signal beyond persistence. Ours does not.

- [ ] **Experiment L — Domain shift at inference** *(supports F2)*
  - Question: the Volatile LSTM is trained on 181 volatile windows but applied to all 1,434 test windows. How does it behave on out-of-distribution (calm) inputs?
  - Result: on calm-dominant test days (n=1,407), Volatile LSTM MSE = far worse than baseline; on strictly-volatile days (n=27), Volatile LSTM (or ensemble) wins directionally but not significantly.
  - **Design critique**: the ensemble's weighting via HMM soft probabilities doesn't fully shield calm days from the volatile LSTM's constant-0.018 output, because the HMM produces small nonzero `p_volatile` even on mostly-calm days near regime boundaries.

- [ ] **Experiment M — Training stochasticity (already noted in discussion.md)**
  - Question: how reproducible are LSTM rankings across runs with the same config?
  - Result: between a previous run and our fresh-clone rerun, ensemble MSE jumped from 1.7e-5 to 2.6e-5 (+53%). Calm LSTM MSE doubled (2.0e-5 → 4.1e-5). Same seeds, same hyperparameters, same data.
  - **Implication**: any single-seed ranking claim is fragile. Multi-seed mean ± std is the right reporting standard.

### 4.4 Ensemble Experiments

- [ ] **Experiment N — Overall ensemble vs baseline (Diebold-Mariano)**
  - Already in nb 06 §6. Results from fresh run (n=1,434):
    - Naive vs Baseline (MAE): **p = 0.014** → Baseline beats Naive on MAE. (MSE: p = 0.17, not significant.)
    - Baseline vs Ensemble (MAE): **p = 0.046** → Baseline beats Ensemble on MAE. (MSE: p = 0.08, marginal.)
    - Baseline vs Ensemble, strictly-volatile subset (n=27): p = 0.54 / 0.83 → can't reject either way, underpowered.
  - **Defensible claims**:
    - ✅ "The baseline LSTM is more accurate than rolling_std persistence (DM MAE p = 0.014)."
    - ⚠️ "Baseline slightly outperforms the HMM-LSTM ensemble on aggregate (DM MAE p = 0.046)." — not "the ensemble fails," which would be overclaiming.
    - ⚠️ "On strictly-volatile days the ensemble is *directionally* better (MSE 5e-5 vs 8e-5) but the n=27 subsample is underpowered." — directional only.

- [ ] **Experiment O — Regime-stratified performance breakdown**
  - Split test set by HMM confidence (`p_volatile > 0.8`). Report MSE/RMSE within each subset.
  - Strictly calm (n=1,407): Baseline wins (MSE 0.00001 vs ensemble 0.00003).
  - Strictly volatile (n=27): Ensemble wins (MSE 5e-5 vs baseline 8e-5).
  - **Visual**: color-coded per-regime MSE table, or two side-by-side bar charts.

- [ ] **Experiment P — Learned linear gate (Ridge stacking meta-learner)** *(headline finding F3)*
  - Question: does replacing the HMM soft-probability weighting with a *learned* linear gate over all three LSTM predictions + HMM probs improve over the HMM ensemble? Over the plain baseline?
  - Setup: Ridge regression on `[baseline_pred, calm_pred, volatile_pred, p_calm, p_volatile] → target`. Fit on train (2,960 rows), α ∈ {0, 0.01, 0.1, 1, 10, 100} swept on val, best model evaluated on test.
  - Result:
    - Best α = 0.1 (val MSE 4.19e-5).
    - Learned coefficients: baseline +0.17, calm +0.36, **volatile −0.002** (zero), **p_calm −0.004** (zero), **p_volatile +0.004** (zero), intercept +0.0077.
    - Test: Gated MSE 1.87e-5 (vs Baseline 1.59e-5 — worse; vs HMM-ensemble 2.61e-5 — better).
    - DM tests: Gated vs Baseline p = 0.57 MAE (**tie**); Gated vs HMM-ensemble p = 0.16 MAE (tie, directionally better).
  - **Two key findings**:
    1. **The learned gate zeros out the Volatile LSTM and the HMM probabilities** — independent confirmation that those inputs carry no exploitable linear signal.
    2. **Even with access to all inputs, the learned gate cannot beat baseline** — the best it can do is tie. Adding more capacity produces a worse predictor.
  - **Also a design note for the paper**: `p_calm + p_volatile = 1` always, so the two columns are perfectly collinear. OLS exploded with intercept ~10⁸; Ridge regularization saved the fit. If we were to do this cleanly, drop one of the two probability columns. Worth a sentence.
  - **Phase 2 (nonlinear MLP gate) was skipped** on these grounds. See Combined framing §F3 for the structural argument: (a) p_volatile rarely varies (97% of days are fully calm), (b) the volatile expert is a constant, so even a perfect gate has nothing useful to gate *toward*. A nonlinear gate faces the same two constraints; no additional structure is discoverable.

### 4.5 Post-sentiment-merge three-way comparison (variants A, H, B)

*Full writeup in `notebooks/07_variant_comparison.ipynb` and the `supplementary/discussion.md` entry dated 2026-04-23 (late-late evening). Supersedes the pre-merge variant-B single-seed experiment in `notebooks/archive/07_variant_b_experiment.ipynb`.*

**Seven LSTMs trained; five evaluable test-set predictors.** From the seven trained models:

| # | Trained by | Checkpoint | Features used | Produces predictor |
|---|---|---|---|---|
| 1 | nb 04 | `lstm_baseline.pt` | 7 (stationary + sentiment) | A baseline (single-LSTM) |
| 2 | nb 05 | `lstm_calm.pt` | 7 | (component of A ensemble) |
| 3 | nb 05 | `lstm_volatile.pt` | 7 | (component of A ensemble) |
| 4 | nb 07 | `lstm_baseline_H.pt` | 8 (+ `p_volatile`) | H baseline (single-LSTM) |
| 5 | nb 07 | `lstm_baseline_B.pt` | 10 (+ VIX family) | B baseline (single-LSTM) |
| 6 | nb 07 | `lstm_calm_B.pt` | 10 | (component of B ensemble) |
| 7 | nb 07 | `lstm_volatile_B.pt` | 10 | (component of B ensemble) |

The 5 evaluable predictors (beyond the naive `rolling_std_21` reference) are **3 baselines** (A, H, B single-LSTM outputs) and **2 soft-probability ensembles** (A, B). Variant H has no ensemble because it *is* a single LSTM that consumes `p_volatile` as a feature — testing whether the regime signal needs the ensemble architecture at all.

- [ ] **Experiment Q — Three-way comparison: regime-split ensemble vs regime-as-feature vs +VIX family**

  - Question 1 (*feature fix for F1*): does adding forward-looking VIX family features pull the persistence-shift lag toward zero?
  - Question 2 (*architectural vs feature exploitation of the HMM signal*): does passing `p_volatile` as an LSTM input feature (variant H) match or beat the soft-probability gated ensemble (variant A)?
  - Setup: all variants share the same HMM regime labels (regime_probabilities.parquet, trained on `HMM_PRICE_VOLUME_FEATURES + [bullish, bearish]`). Only the LSTM input set changes across variants.

  **Headline metrics (n = 1,434 common test rows):**

  | Predictor | MSE | MAE | F1 peak lag | F1 peak r | F1 r@lag0 |
  |---|---|---|---|---|---|
  | naive (rolling_std_21) | 2.22×10⁻⁵ | 0.0031 | −21 | 1.000 (identity) | 0.458 |
  | **A ensemble** | **1.42×10⁻⁵** | **0.0025** | −16 | 0.808 | **0.600** |
  | **H baseline** (+`p_volatile`) | 1.58×10⁻⁵ | 0.0025 | −16 | 0.855 | 0.502 |
  | **B ensemble** (+ VIX) | **1.40×10⁻⁵** | **0.0025** | −17 | 0.815 | 0.541 |

  **DM pairwise (h=21, Bartlett HAC, HLN-corrected):**

  | Comparison | MSE p | MAE p | Verdict |
  |---|---|---|---|
  | naive vs A.ensemble | 0.09 | **0.002** | A wins MAE |
  | naive vs H.baseline | 0.13 | **0.006** | H wins MAE |
  | naive vs B.ensemble | 0.08 | **0.002** | B wins MAE |
  | A vs H | 0.35 | 0.75 | **tied** |
  | A vs B | 0.79 | 0.98 | **tied** |
  | H vs B | 0.27 | 0.73 | **tied** |

  **Answer to Q1 (feature-fix for F1)**: no. Peak cross-correlation lag stays at −16 (A, H) and −17 (B) — VIX features pull the lag *backward* by one day, not forward. F1 survives.
  **Answer to Q2 (gating vs feature)**: tied. A and H are statistically indistinguishable on MSE/MAE (p=0.35/0.75). The regime signal is equally exploitable via gating (A) or as a feature (H); neither architectural choice adds value over the other. Both have drawbacks: A risks F2 failures in the volatile LSTM; H bypasses F2 entirely but has lower zero-lag correlation (0.502 vs 0.600).

  **F2 diagnostic under all three variants**: volatile-LSTM pathology flips seed-dependently between modes. Variant A's volatile LSTM collapsed to a near-constant output (std 1.9×10⁻⁴, range 1.5×10⁻³). Variant B's volatile LSTM instead produced wild outputs (std 0.65, max 2.71 — 271 % daily vol). The soft-prob ensemble gating limits damage in both cases because `p_volatile` is small on ~97 % of days, but the volatile LSTM contributes no reliable signal in either variant. Variant H *avoids F2 by design* (no separate volatile LSTM) — that's a practical argument for variant H even without an MSE edge.

  **Clean claims for the paper:**
  1. ✅ *"All three learned variants (regime-split ensemble, HMM-as-feature baseline, VIX-augmented ensemble) are statistically indistinguishable on MSE and MAE at h = 21 (DM pairwise p ≥ 0.27). Once the baseline has sentiment, neither forward-looking implied-volatility data nor direct ingestion of the HMM regime probability provides a detectable improvement."*
  2. ✅ *"The cross-correlation peak lag stays at −16 / −17 days in every variant. The persistence-shift floor is a property of the daily-frequency × 21-day-forward target, not the feature set or the architectural choice."*
  3. ✅ *"The volatile-regime LSTM fails in two distinct seed-dependent modes (near-constant vs wild output). This seed-sensitivity is itself evidence that F2 is a data-scarcity problem, not a feature-set or hyperparameter problem — Optuna lands in different pathological optima on different runs because there's no stable useful solution to find on ~180 volatile training windows dominated by a single crisis."*

  **Caveat (training-set size asymmetry)**: variant A's LSTMs see the full 3,020 training rows; variant B's see ~2,034 (VXVCLS starts 2007-12); variant H's see ~3,000 (~710 NaN at the start where the HMM didn't produce probabilities). Test windows identical; this asymmetry is minor but worth a methodology sentence.

---

## General Don't-Miss Items

### Strengths to highlight (pick at least one per Section 5)

- [ ] **Rigorous multi-angle diagnosis of *why* the regime-split didn't help** — cross-correlation lag analysis (F1) + prediction-variance analysis (F2) + learned-gate confirmation (F3) independently converge on the same structural conclusion. Three complementary pieces of evidence, not just one.
- [ ] **Tested and falsified the "the gating is wrong" hypothesis.** By replacing HMM-soft-probability weighting with a learned Ridge meta-learner and seeing it *also* fail to beat baseline, we ruled out the possibility that the HMM probabilities were the problem. The Ridge coefficients zeroing out both the Volatile LSTM and the HMM probs is direct evidence that the issue is the model/data pairing, not the weighting mechanism.
- [ ] **Scope discipline — deliberate restriction to price/volume features.** We consciously excluded canonical forward-looking features (implied vol / VIX, option-positioning, scheduled-event calendars) that most vol-forecasting papers include. This lets us characterize the *structural limits* of models operating on return-based inputs alone — a cleaner research claim than "we used all available features and didn't beat baseline." Frame in the intro: *"We study the structural limits of regime-aware volatility forecasting on price/volume data."*
- [ ] **Economic sanity check, not just statistical fit**: volatile state aligns with 2008 GFC, 2020 COVID; 0 false positives in benign 2016–2019 window.
- [ ] **Defense against degenerate EM solutions** (smart init + penalized LL) — addresses a real and well-known failure mode.
- [ ] **Multi-test triangulation** for normality — more robust than any single test.
- [ ] **Stationarity fix (Phase 6)** — discovered and fixed a real distributional drift problem; MAPE went from ~600% to ~25%.
- [ ] **Significance testing added** (Diebold-Mariano, nb 06 §6) — rankings are statistically backed, not just point estimates.
- [ ] **Reproducibility trail**: every discrepancy from the proposal is logged in `discussion.md` with a rationale.

### Weaknesses to acknowledge (required by Section 5)

- [ ] **Central negative result — the regime split doesn't help and all LSTMs collapse to persistence** (F1, F2, F3, F4). Frame as a *structural* finding, not an implementation failure: given rare regimes × daily frequency × 21-day horizon, the effective number of independent volatile training events is single-digit, so per-regime deep models cannot learn regime-specific dynamics from this data.
- [ ] **Volatile LSTM is structurally degenerate** (F2) — 181 training windows span only ~4–5 independent volatile events (GFC-dominated); the model collapses to outputting its training mean (constant ~0.0179). Root cause: data scarcity (F4), not hyperparameter choice.
- [ ] **Both regime LSTMs are reactive, not predictive** — shift of ~17 days measured via cross-correlation. Any deep model on 21-day-forward vol with price/volume features alone is bounded by persistence.
- [ ] **HMM regime detection is also reactive** — P(volatile) only spikes *after* volatile returns enter the observation window. The ensemble gate therefore can't pre-position to the volatile LSTM before the transition.
- [ ] **Sentiment + put/call scaffolded but never activated** — divergence from proposal; Ho et al. and Gupta et al. motivation is weakened. This is the single biggest reason we have no forward-looking signal.
- [ ] **Single-seed LSTM results are noisy** — same config produced ensemble MSE 1.7e-5 in one run and 2.6e-5 in another (+53%). Multi-seed mean ± std needed for robust claims.
- [ ] **Volatile test subset is underpowered** — n=27 strictly-volatile days; DM tests can't reject despite large point-estimate gaps.
- [ ] **Scope narrowing to SPY only** — doesn't test generalization across assets (XEQT/MAG7 from the proposal).
- [ ] **No formal stationarity test** (ADF) in EDA — Phase 6 drift caught empirically rather than diagnosed upfront.
- [ ] **1/m vs 1/(m-1) in the target formula** — clarify or cite explicitly.

### Future work (Section 5 requires at least some)

Ordered by leverage — the three data-scarcity axes (F4) first, then lower-leverage items:

- [ ] **[Data axis 1] Higher-frequency data** — shift to intraday (5-minute) bars. 78× more observations per day; volatile *windows* become plentiful even with the same calendar history. Requires rethinking the target (intraday vol ≠ daily realized vol) and handling microstructure noise, but addresses the single-digit-volatile-events bottleneck directly. *Highest potential impact.*
- [ ] **[Data axis 2] Extend history pre-2004** — SPY inception is 1993; can go back to ~1994. Picks up 2001 dot-com crash and LTCM 1998. Doubles training horizon, but the ~4% volatile base rate is structural, so effective independent volatile events only grows from ~5 to ~10. Useful but not a game-changer.
- [ ] **[Data axis 3] Transfer learning / asset pooling** — with an important caveat: *naive* pooling fails. Because SPY already contains most large US equities, and systemic crises (2008, 2020) hit all risk assets simultaneously, pooling SPY with QQQ/IWM/individual US stocks *duplicates* the same volatile events rather than adding independent ones. The data-augmentation win is specifically in **non-systemic** volatile episodes. Taxonomy by independence from SPY (Level A → D, decreasing similarity to source):
  - **Level A — Regional equity indices** *(same asset class, regional crises)*: Euro STOXX / EZU → Euro debt crisis 2010-12, Brexit, 2022 energy crisis. Nikkei / EWJ → Fukushima 2011. Hang Seng / FXI → 2015 CNY crash, 2021-22 property bust. EM (EEM / VWO) → taper tantrum 2013, Turkey 2018. Pros: same dynamics; trivial target transfer. Cons: ~0.4-0.8 correlation with SPY; spillover during global events.
  - **Level B — Asset-class independence** *(different risk factors)*: TLT → rate vol (1994, 2013, 2022, UK gilts). HYG → credit vol. GLD → safe-haven vol. USO → commodity supply shocks (2014-16, 2020 negative oil). DXY → FX vol. Pros: genuinely orthogonal crises; many more independent events. Cons: vol dynamics may differ by asset class — transfer assumption needs ablation validation.
  - **Level C — Sector ETFs** *(partial, sectors are inside SPY)*: XLE 2014-16 oil crash, XLF 2008 banking, XLK 2022 tech rate-hike. Pros: cheap to add; same dynamics. Cons: highly correlated with SPY — small marginal gain.
  - **Level D — Synthetic augmentation** *(bypass data scarcity entirely)*: GARCH-simulated regimes; block bootstrap of GFC days; noise augmentation on volatile windows. Pros: unlimited; tunable base rate. Cons: model learns synthetic statistics, not market dynamics — becomes a separate research project requiring its own validation.
  - **Recommended priority for this project**: Level A (low risk, direct gain) → Level B (higher ambition, needs ablations) → Level C (supplementary) → Level D (separate project).

- [ ] **Transferability of volatile-regime dynamics across asset classes** — standalone companion contribution worth considering. Before investing in transfer learning, compute autocorrelation + tail-index profile of volatile days per asset class and measure similarity. If volatile dynamics are universal → transfer-learning path is defensible. If asset-specific → second-order finding: "volatile-regime dynamics do not generalize across asset classes; data-scarcity cannot be solved by pooling." Either outcome is a paper-worthy contribution in its own right.
- [ ] **Activate sentiment + put/call features** (proposal's Ho et al. / Gupta et al. motivation) — orthogonal to the data-scarcity axes; addresses the persistence-predictor problem (F1) by introducing non-price signals the baseline LSTM currently lacks. Single most likely change to yield a *forward-looking* signal.

- [x] **Add options-derived forward-looking features** — *tested as variant B (see §4.5)*. Added `vix` (FRED `VIXCLS`), `vix_log_change`, and `vix3m_minus_vix` (`VXVCLS` − `VIXCLS`) to the LSTM feature set; retrained baseline + calm + volatile; re-ran ensemble. **Result post-sentiment-merge: variant B ensemble ties with variant A ensemble (DM p=0.79 MSE, p=0.98 MAE) and peak cross-correlation lag stays at −17. F1 survives even with explicit forward-looking implied-vol input.** Remaining sub-items:
  - **Add SKEW and VVIX** — FRED discontinued these series; yfinance's `^SKEW` / `^VVIX` were too unreliable to depend on. A reliable data source (paid CBOE DataShop or scraped archive) would let us test whether tail-risk premium (SKEW) or vol-of-vol (VVIX) carry forward-looking signal that spot VIX does not.
  - **Tier 2 — IV skew and risk reversal** (requires SPX option chain data, more engineering): IV(25Δ put) − IV(25Δ call) captures the asymmetric tail premium that aggregate put/call volume blurs. Variant B's evidence suggests this is a low-priority lever — if spot VIX only moves the shift by 1-3 days, finer option structure is unlikely to move it dramatically more.
  - **Tier 3 — Dealer positioning (GEX, vanna)**: specialized paid data (SpotGamma / Squeezemetrics); likely out of scope.
- [x] **HMM-as-feature architecture (variant H)** — *tested as variant H in §4.5*. Fed `p_volatile` to a single baseline LSTM instead of using it to gate an ensemble. **Result: H ties with A (DM p=0.35 MSE, p=0.75 MAE) and avoids F2 by design (no separate volatile LSTM).** Not a clear win on metrics but structurally simpler and F2-immune.
- [ ] **Variant O — "pre-everything" OHLCV-only baseline** — retrain HMM on `HMM_PRICE_VOLUME_FEATURES` (11 feats, no sentiment) and LSTMs on `LSTM_STATIONARY_FEATURES` (5 feats, no sentiment / VIX / p_volatile) to establish the lower bound "floor" the paper cites. Would extend the §4.5 comparison table from 4 rows (naive + A + H + B) to 5 rows (+ O) and answer "did any of our post-phase-6 additions actually move the needle past where we started?". Requires 1 HMM retrain + 3 LSTM trainings + inline variant-O ensemble (~25-40 min).
- [ ] **Multi-seed study (3 seeds per LSTM config)** — every metric reported in §4.5 is a single-seed draw from a noisy distribution. Given the F2 seed-dependent pathology (different runs flip between "constant" and "wild" volatile-LSTM modes), proper mean ± std reporting across 3 seeds per config is the correct rigour level for a published paper. Planned for Google Colab rather than local CPU (parallelize across tabs).
  - **Crucial benchmark caveat when any of these are added**: the relevant baseline shifts from `rolling_std_21` to **VIX (or VIX-derived blend) alone as a direct prediction**. If `LSTM + VIX ≈ VIX_alone` on test, the LSTM contributes nothing; the signal was already in VIX. Beating VIX directly is a genuinely hard and interesting result.
  - **Distinction from your partner's work**: put/call volume ratio is *sentiment-like* (counting flow); VIX / SKEW / VVIX are *expectation-like* (prices of future-outcome contracts). Complementary rather than substitutes. A strong version of the non-price-feature pipeline would include both.
  - **Why we excluded it for this paper**: including VIX would muddy the current clean negative result ("price-only features cannot predict forward vol"). A reviewer would correctly attribute improvements to VIX rather than to our architecture. Cited as future work keeps the current research claim tight.

- [ ] **Scheduled-event / calendar features** — FOMC meeting dates, CPI / NFP release dates, earnings-season boundaries, options expiration dates. Genuinely forward-looking (on Monday you *know* Wednesday is the Fed meeting). Typically adds small but statistically significant power to vol forecasts.
- [ ] **Clip or gate volatile predictions at low p_volatile.** Simple fix: only use the volatile LSTM when p_volatile > some threshold; otherwise use baseline directly. Prevents the constant-0.018 output from contaminating calm days. Low-impact band-aid — doesn't address root cause, but simple.
- [ ] **Replace the volatile neural net with a non-degenerate simpler model** (e.g., always output `rolling_std_21` for the volatile regime). Less glamorous but avoids the constant-collapse failure mode. Valid ablation for the paper.
- [ ] **Multi-seed LSTM robustness study** (k ≥ 5 seeds, report mean ± std). Already open item #4 in `discussion.md`. Essential before locking any ranking claim.
- [ ] ~~**Learned gating network**~~ — **attempted in Phase 1 (linear Ridge stacking); also failed** (F3). Zeroed out Volatile LSTM and HMM probs. Nonlinear Phase 2 skipped on structural grounds.
- [ ] **Extend test window pre-2020** to include GFC — dramatically increases the volatile-day sample size for DM tests.
- [ ] **Extend to other assets** (XEQT, MAG7) for cross-asset generalization per the original proposal.

### Figures to include (prioritize over tables when possible)

- [ ] **Cross-correlation lag plot** for each model vs target (one panel per model, correlation as a function of lag, peak marked). Immediately visualizes finding F1.
- [ ] **Prediction time series overlay**: target, baseline, and baseline-shifted-by-16-days — showing the shift collapses at the right lag.
- [ ] **Prediction variance/range histogram** per model — immediately shows the Volatile LSTM's collapse to a near-point mass around 0.018 (finding F2).
- [ ] SPY regime overlay (shaded volatile periods on price chart + P(vol) panel).
- [ ] HMM model-comparison bar chart (val LL / BIC / AIC across Gaussian vs GMMHMM(2) vs GMMHMM(3)).
- [ ] Volatility clustering visual (|log_return| over time + ACF of r²).
- [ ] Target distribution: raw vs log-transformed realized vol (2-panel histogram).
- [ ] Per-regime return histograms (calm vs volatile) with Levene p-value annotation.
- [ ] Transition matrix heatmap for the winning model.
- [ ] Regime-stratified MSE bar chart (strictly-calm vs strictly-volatile subsets, baseline vs ensemble).

### References to cite (HMM/data side)

- [ ] Andersen & Bollerslev (1998) — realized vol formulation for the target.
- [ ] Cont (2007) — volatility clustering stylized fact.
- [ ] Rabiner (1989) — HMM algorithms.
- [ ] Zhang et al. (2019) — GMMHMM for financial time series.
- [ ] Nystrup, Hansen & Madsen (2015–2019) — penalized EM for regime-based asset allocation.
- [ ] Lee (2017) — higher-order HMM via state augmentation.
- [ ] Ho et al. (2012) — put/call volume and vol *(mention even though we didn't use it, to motivate future work)*.
- [ ] Gupta et al. (2023) — sentiment and realized vol *(same)*.
- [ ] Whaley (1993) or Whaley (2000) — original VIX construction and interpretation *(cite in Related Work when explaining what forward-looking options-derived features exist)*.
- [ ] Corsi (2009) — HAR-RV model using multi-horizon realized vol *(canonical benchmark in the vol-forecasting literature; worth citing as contrast)*.
- [ ] Christensen et al. (2023) — the machine-learning approach to vol forecasting cited in the proposal; explicitly contrasts with older econometric approaches and discusses implied-vol features.

---

**Status:** sections 3.1, 3.2, 3.3, 3.4 and 4.1-4.4 populated. Headline findings F1 and F2 surfaced at top. Remaining: confirm final figures, verify Diebold-Mariano p-values against re-runs before freezing claims.
