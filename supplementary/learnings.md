HMM_N_MIX [1,2,3] :

Good pushback — this is the key mechanic to understand. The two (or three) Gaussians in a mixture are not duplicates. They learn different means and covariances, so they    
  capture distinct sub-clusters within a state.                                                                                   

  Mathematically — a single-Gaussian emission for state s is:                                                                                                                  
   
  p(o | s) = N(o | μ_s, Σ_s)                                                                                                                                                   
                                                                                                                                                                               
  One mean, one covariance, one bell curve. That's it.                                                                                                                         
                                                                                                                                                                               
  A GMM mixture emission with K components is:                                                                                                                                 
                  
  p(o | s) = π_{s,1} · N(o | μ_{s,1}, Σ_{s,1})
           + π_{s,2} · N(o | μ_{s,2}, Σ_{s,2})                                                                                                                                 
           + ...                                                                                                                                                               
                                                                                                                                                                               
  K different means, K different covariances, weighted together. EM explicitly learns each component to take ownership of a different slice of the state's observations — it   
  can't converge to two identical Gaussians because each one has to "claim" observations the other doesn't fit as well.
                                                                                                                                                                               
  Why would you want this? Because within a single regime, observations can be multi-modal. Your own trained model (nb 03 cell 20) proves this:                                
  
  State 0 (calm) — mix_0 vs mix_1:                                                                                                                                             
  mix_0:  log_return=-0.61, oc_return=-0.53  → "calm-down days"
  mix_1:  log_return=+0.51, oc_return=+0.43  → "calm-up days"                                                                                                                  
  Calm days split into small-decline and small-gain subclusters. Opposite signs, both low-vol.                                                                                 
  
  State 1 (volatile) — mix_0 vs mix_1:                                                                                                                                         
  mix_0:  abs_return=+0.29, rolling_std_21=+2.00  → "mild volatile" (elevated but not extreme)
  mix_1:  abs_return=+2.48, rolling_std_21=+3.55  → "extreme volatile" (crisis-level)                                                                                          
  Volatile days split into "elevated risk" vs "full-blown crisis."                                                                                                             
                                                                                                                                                                               
  A single Gaussian for state 1 would have to fit one smooth bell over both mild and extreme volatile days — producing a distribution that's a bad fit for both (too wide for  
  mild, too narrow for extreme). The mixture nails both.                                                                                                                       
                                                                                                                                                                               
  On "two Gaussians converging to the same value": EM can in theory do that if you initialize them identically and don't break the symmetry. That's why _smart_init_hmm in     
  hmm_model.py adds small random offsets (rng.normal(0, 0.05, n_features)) to each mixture component's initial mean — it's deliberately preventing this degeneracy. Once EM
  runs, the components drift apart to fit different subclusters.                                                                                                               
                  
  So n_mix=2 asks the model: "is each regime actually better described as two sub-patterns?" Here the answer was yes — and BIC confirmed the extra parameters paid off. 

---

HMM config knobs — HMM_N_ITER, HMM_RANDOM_STATE, HMM_N_RESTARTS, HMM_COVARIANCE_TYPE, HMM_DURATION_PENALTY, HMM_MARKOV_ORDERS:

  HMM_N_ITER = 200 — max iterations of the EM (Baum-Welch) algorithm per fit. Each iteration does an E-step (compute state posteriors) + M-step (update transition/emission params). If EM hasn't converged within 200, hmmlearn stops and logs a "Model is not converging" warning.

  HMM_RANDOM_STATE = 42 — base seed for reproducibility. Passed to hmmlearn's random initialization. Each restart uses random_state + restart_idx so every restart gets a different but deterministic seed.

  HMM_N_RESTARTS = 50 — EM only finds *local* optima. You start from 50 different initial parameter settings, run EM to convergence from each, and keep the best-scoring result. Restart 0 uses the smart volatility-quantile init (_smart_init_hmm); restarts 1-49 use hmmlearn's default random k-means init. 50 restarts ≈ enough to reliably find a good solution for this problem size.

  HMM_COVARIANCE_TYPE = "full" — each state's Gaussian has a full 11×11 covariance matrix (66 free entries). Alternatives: 'diag' (features independent within state), 'spherical' (single variance), 'tied' (shared across states). 'full' is most expressive but adds the most parameters — BIC handles that trade-off.

  HMM_DURATION_PENALTY = 3000.0 — λ in the penalised-LL used for restart selection. Discourages degenerate rapid-switching solutions. Rule of thumb: set ≈ n_training_obs (3,000 here) so the penalty is comparable in magnitude to the log-likelihood.

  HMM_MARKOV_ORDERS = [1, 2] — declares which Markov orders to test in Phase 2b. hmmlearn only supports 1st-order natively; 2nd-order is approximated via state-space or feature-space augmentation (both covered in nb 03).

---

EM vs Viterbi vs forward-backward — is EM the same as finding the most likely sequence?

  No — they're different algorithms solving different problems. In HMM-land there are three famous algorithms and it's easy to conflate them:

  ┌──────────────────┬──────────────────────────────────────────┬────────────────────────────┬────────────────────────────────────────────────────────────────────────────┐
  │    Algorithm     │               What it does               │           Input            │                                   Output                                   │
  ├──────────────────┼──────────────────────────────────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ EM (Baum-Welch)  │ Learns the model parameters              │ Observations + guessed     │ Trained transition matrix, means, covariances                              │
  │                  │                                          │ init                       │                                                                            │
  ├──────────────────┼──────────────────────────────────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ Viterbi          │ Finds the single most likely state       │ Observations + trained     │ One hard label per timestep (e.g., [calm, calm, volatile, volatile, calm,  │
  │                  │ sequence                                 │ model                      │ ...])                                                                      │
  ├──────────────────┼──────────────────────────────────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ Forward-backward │ Computes soft probabilities at each step │ Observations + trained     │ P(state_t = s | all obs) — a distribution per timestep                     │
  │                  │                                          │ model                      │                                                                            │
  └──────────────────┴──────────────────────────────────────────┴────────────────────────────┴────────────────────────────────────────────────────────────────────────────┘

  How they relate in our project:
  - model.fit(X) in hmm_model.py runs EM → produces hmm_winner.joblib (a fitted GMMHMM).
  - model.decode(X) runs Viterbi → produces the viterbi_state column in regime_probabilities.parquet (used to decide which LSTM to train each day on).
  - model.predict_proba(X) runs forward-backward → produces p_calm and p_volatile (the soft weights for Phase-4 ensemble mixing).

  Subtle link: EM uses forward-backward internally during each E-step — it needs soft state posteriors to do the M-step. But EM itself isn't "find the sequence," it's "find the parameters." Once you have the trained model, then you run Viterbi or forward-backward to actually extract sequences or probabilities.

---

HMM_RANDOM_STATE = 42 — why this value? what does it actually do?

  It's a reproducibility knob, not a quality knob. The value itself is meaningless.

  What it actually is: a seed for the pseudorandom number generator (PRNG). Computers don't produce true randomness — they produce a deterministic sequence that *looks* random, starting from a seed. Same seed → same sequence, every time, forever.

  Why HMM training needs randomness at all:
  - hmmlearn's default init uses k-means, which picks random starting centroids.
  - Our _smart_init_hmm adds small Gaussian noise (rng.normal(0, 0.05, ...)) to differentiate mixture components so EM doesn't collapse them.
  - The 50 restarts use *different* random seeds to explore different starting points in parameter space — if they all started identically, there'd be no point running 50.

  Specifically in hmm_model.py:
    for restart_idx in range(n_restarts):
        seed = random_state + restart_idx   # 42, 43, 44, ..., 91
        model = hmm.GMMHMM(..., random_state=seed)

  So each restart gets a distinct but fully determined seed. Fifty restarts, fifty different random-feeling initializations, all reproducible.

  Why "42" specifically: it's a coder's inside joke from Hitchhiker's Guide to the Galaxy ("the answer to life, the universe, and everything"). You could set it to 0, 1, 7, or 1337 — the HMM would train equally well, just produce a numerically different (but statistically equivalent) final model.

  Why set it at all (vs leave it unset and let Python pick a truly random seed)? Reproducibility. With random_state=42, re-running the notebook tomorrow gives the *exact same* HMM as today — same transition matrix, same regime means, same Viterbi labels. Without it, you'd get a slightly different model each run and couldn't debug or compare fairly. Standard practice in any ML code.

  Caveat we hit earlier: PyTorch on CPU isn't fully seeded by default even with random_state=42 — that's why LSTM runs still differ across machines/runs. HMM training via hmmlearn is much more deterministic than LSTM training.

---

HMM_DURATION_PENALTY = 3000 — what does it explicitly penalize? is 3000 appropriate?

  What it penalizes: short average regime durations — models where the hidden states switch rapidly back and forth instead of representing sustained calm/volatile periods.

  The math (from src/hmm_model.py _penalised_score):
    penalised_LL = log P(O | model) − λ · Σ_s (1 − p_stay_s)

  where p_stay_s = transmat_[s, s] (probability of staying in state s). Two useful identities:
  - avg_duration_s = 1 / (1 − p_stay_s)  (expected time in state s; geometric distribution)
  - So (1 − p_stay_s) = 1 / avg_duration_s

  The penalty term is the sum of inverse durations across all states.

  Concrete magnitudes (our 2-state HMM):

  ┌─────────────────────────────────────┬──────────────────┬─────────────────┬──────────────────────────────┐
  │ Scenario                            │ p_stay per state │ Σ(1 − p_stay)   │ Penalty contribution at λ=3000│
  ├─────────────────────────────────────┼──────────────────┼─────────────────┼──────────────────────────────┤
  │ Degenerate (flips every 2 days)     │ ~0.5, 0.5        │ ≈ 1.0           │ 3,000                        │
  │ Our winning GMMHMM                  │ 0.996, 0.919     │ ≈ 0.086         │ 258                          │
  │ Very sticky (durations = 500 days)  │ 0.998, 0.998     │ ≈ 0.004         │ 12                           │
  └─────────────────────────────────────┴──────────────────┴─────────────────┴──────────────────────────────┘

  So a degenerate solution pays a penalty ~2,742 units higher than our healthy one. The penalty selects the healthy model only if the degenerate model's raw LL advantage is less than 2,742.

  Is λ = 3000 appropriate?

  The rule of thumb in the code comment is "λ ≈ n_training_observations" — here n_train = 3,000 after dropping NaN. That makes the penalty comparable in magnitude to the log-likelihood, which is the key scale-matching choice.

  Pros of our setting:
  - Sized correctly for this dataset (3,000 obs). If you had 30,000 obs, you'd want λ ≈ 30,000, because larger samples let the LL grow linearly.
  - It reliably filters the degenerate fast-switching solutions we were hitting on random k-means inits (nb 03 shows these in the "Model is not converging" warnings).
  - Our winning model has volatile avg-duration of 12 trading days, which is economically reasonable (crisis episodes last 2-4 weeks).

  Caveats / things a reviewer could push on:
  1. It's one-sided — linearly biases toward longer durations. If the true volatile regime should actually be brief 3-5-day bursts, this penalty still charges them (contribution ≈ 600-1000). A thresholded penalty (e.g., only penalize when p_stay < 0.8) would be more principled but less standard.
  2. Reference (Nystrup et al.) uses this formulation for regime-based asset allocation on daily returns. Reasonable citation support, but it's a rule of thumb, not a derived optimum.
  3. No cross-validation on λ — we set it once and didn't sweep. Could be worth showing in the paper: "we verified results are robust under λ ∈ [1000, 10000]" as a sensitivity check, otherwise a skeptical reviewer will ask.

  Short answer: λ = 3000 is a defensible, well-scaled choice for this data, but it's an ad-hoc regularizer and deserves either a sensitivity analysis or an explicit citation in the paper's methodology section.

---

Why do LSTM_BASELINE_FEATURES and HMM_FEATURES differ?

  Two distinct reasons, both rooted in how each model processes time.

  Reason 1: Different jobs, different information needs.

  The HMM sees one observation at a time — no memory, no sequence. Each day's feature vector must be a complete snapshot of "what does today look like?" That's why HMM_FEATURES includes rolling_mean/std_{5,10,21} — the HMM has no access to the previous 21 days of returns, so you bake the recent context directly into today's observation.

  The LSTM sees a sliding window of seq_len days (21 or 42). It already has access to the raw history through its recurrent architecture — you don't need to pre-compute rolling statistics because the LSTM can learn them internally from the sequence of log_returns if they're useful. Handing it rolling_std_21 as a feature would be redundant information the LSTM already implicitly has.

  Reason 2: Stationarity (Phase 6 decision) hit the LSTM hard.

  SPY volume has grown from ~10M shares/day in 2004 to ~100M+ today. So:
  - log_volume ranges roughly 16 → 20 over the dataset — a clear non-stationary drift.
  - relative_volume_21d = Volume / 21-day avg Volume centers around 1.0 at every point in time, regardless of era.

  The HMM is fine with log_volume: it's scaled with a StandardScaler fit on train data, and the HMM just uses whatever shows up in each timestep's emission density. A high log_volume reading in 2026 simply means "this day has lots of volume," which the model interprets contextually.

  The LSTM got burned by log_volume. The plan doc documents this as Phase 6: with 2026 scaled values far outside the 2004-2015 training range, activations blew up and predictions drifted permanently upward (MAPE ~600%). Swapping log_volume → relative_volume_21d fixed it (MAPE dropped to ~25%). LSTMs are much more sensitive to non-stationary inputs than HMMs because predictions propagate nonlinearly through the network.

  Quick side-by-side:

  ┌──────────────────┬─────────────────────────────────────────────────┬──────────────────────────────────────────┐
  │                  │ HMM (11 feats)                                  │ LSTM baseline (5 feats)                  │
  ├──────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────┤
  │ Volume           │ log_volume (non-stationary, fine for Gaussian)  │ relative_volume_21d (stationary)         │
  │ Rolling stats    │ rolling_mean/std_{5,10,21} — injects context    │ none — LSTM infers from sequence         │
  │ Temporal memory  │ none within feature vector                      │ built into architecture                  │
  └──────────────────┴─────────────────────────────────────────────────┴──────────────────────────────────────────┘

  Consequence for the paper: there's a defensible methodological reason for the asymmetry, not just ad-hoc. The HMM needs context as features; the LSTM needs stationary inputs. Worth one sentence in the methods section so reviewers don't read the difference as a mistake.

---

LSTM config knobs — LOG_TRANSFORM_TARGET, LSTM_PATIENCE, LSTM_RANDOM_STATE, LSTM_N_TRIALS, LSTM_TUNE_EPOCHS, LSTM_FINAL_EPOCHS, LSTM_SEARCH_SPACE:

  LOG_TRANSFORM_TARGET = True
  Train the model to predict log(realized_vol_21d) instead of raw realized_vol_21d. The raw target is right-skewed (skew=3.5, kurtosis=17.4 per the config comment) — taking the log compresses the heavy tail into a near-Gaussian shape (skew=0.75, kurt=1.07), which MSE loss handles much better. At eval time predictions are exp()-ed back so metrics are in the natural volatility scale. Without this, a handful of crisis days dominate training.

  LSTM_PATIENCE = 10
  Early stopping tolerance. After each training epoch, check val MSE. If val MSE hasn't improved for 10 consecutive epochs, stop training and restore the best-seen weights. Prevents overfitting to noise once validation plateaus — otherwise a 100-epoch run would keep pushing training loss down while val loss creeps up.

  LSTM_RANDOM_STATE = 42
  Same concept as HMM_RANDOM_STATE — seed for PRNG-controlled components: Optuna trial sampling, PyTorch weight initialization (partially — CPU PyTorch has known non-determinism). Reproducibility knob.

  LSTM_N_TRIALS = 20
  Optuna will sample 20 hyperparameter configurations total during the search. More trials = better chance of finding a good config, but linear in compute. 20 is a common "thorough but not exhaustive" default for TPE-sampled spaces this size.

  LSTM_TUNE_EPOCHS = 40 (per trial, during Optuna search)
  Each Optuna trial trains for at most 40 epochs (with early stopping). Shorter than the final retrain — 40 is enough to rank hyperparameter configs against each other without fully converging each one. Saves ~60% of search compute vs training each trial to 100 epochs.

  LSTM_FINAL_EPOCHS = 100 (for the final retrain with best params)
  Once Optuna picks the winning hyperparameters, the model is retrained from scratch with those params for up to 100 epochs (with early stopping). More generous epoch budget because you're only doing it once and want full convergence.

  LSTM_SEARCH_SPACE — what Optuna explores:

  ┌────────────────┬────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │ Hyperparameter │ Range / choices            │ What it controls                                                         │
  ├────────────────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ hidden_size    │ {32, 64, 128}              │ Number of units in each LSTM layer. Larger = more capacity, more         │
  │                │                            │ overfitting risk.                                                        │
  │ n_layers       │ {1, 2, 3}                  │ Depth of stacked LSTMs. Deeper = more abstract temporal patterns,        │
  │                │                            │ harder to train.                                                         │
  │ dropout        │ uniform(0.0, 0.5)          │ Dropout prob between LSTM layers. Regularization — turns off some        │
  │                │                            │ units per forward pass.                                                  │
  │ lr             │ log-uniform(1e-4, 1e-2)    │ Adam learning rate. Log-uniform because the good region is               │
  │                │                            │ multiplicative, not additive.                                            │
  │ batch_size     │ {32, 64, 128}              │ Samples per gradient update. Larger = smoother gradient, less            │
  │                │                            │ stochasticity.                                                           │
  │ seq_len        │ {21, 42}                   │ Sliding-window length (how many past days the LSTM sees to predict      │
  │                │                            │ forward vol). 21 ≈ 1 month, 42 ≈ 2 months.                               │
  └────────────────┴────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  Compute math: 20 trials × 40 epochs + 1 × 100 epochs = ~900 total epochs per LSTM. That's why nb 04/05 takes ~an hour each even on a small network.

---

features.py audit — target formula interpretation and minor nits:

  What the proposal literally says (the garbled formula t=1m-1j=1m(rt+j-rt)2): the m-1 in there most likely indicates sample standard deviation (divide by m-1) — the standard unbiased estimator. Plan doc separately quotes it as sqrt(Σ(r_{t+j} - r_bar)²) with no denominator — a third possible reading.

  So there are three interpretations floating around:

  ┌───────────────────────────────────┬───────────────────────────────────────┬──────────────────────┐
  │              Version              │                Formula                │ Scale vs features.py │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────┤
  │ features.py (population std, 1/m) │ $\sqrt{\frac{1}{m}\Sigma(\cdot)^2}$   │ baseline             │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────┤
  │ Sample std (1/(m-1))              │ $\sqrt{\frac{1}{m-1}\Sigma(\cdot)^2}$ │ +2.4% larger         │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────┤
  │ No denominator                    │ $\sqrt{\Sigma(\cdot)^2}$              │ ×√21 ≈ 4.58× larger  │
  └───────────────────────────────────┴───────────────────────────────────────┴──────────────────────┘

  Does it matter for the ML pipeline? No — the model learns to predict whatever target scale is given. Train/val/test are all on the same formula, so metrics are internally consistent. The baseline LSTM and ensemble all see the same target.

  Does it matter for the paper? Yes, a reviewer could pick at it. Three paths:
  1. Clarify with your partners which formula the proposal intended (probably 1/(m-1) given the m-1 in the garbled text).
  2. If it's the 2.4% difference, state (1/m) version explicitly in methods and cite Andersen-Bollerslev — negligible either way.
  3. If the proposal actually meant the no-denominator version, that's a legitimate divergence worth logging in discussion.md under the discrepancy table.

  The code is internally consistent (docstring at line 131 matches the code), so this is a definitional alignment with the proposal issue, not a bug.

  Minor nits (not blockers):

  1. time_split has dead code (lines 188-190) — does a buggy string-slice split, then overwrites with the correct boolean-mask version (193-195). Harmless but confusing; could be cleaned up.
  2. intraday_range uses log((H-L)/Close). Some papers normalize by Open or (O+C)/2 instead. Just a choice, not wrong.
  3. ddof inconsistency: add_rolling_stats uses pandas' .std() default of ddof=1 (sample std), but add_realized_volatility uses .mean() (equivalent to ddof=0). So rolling_std_21 and realized_vol_21d aren't using the same denominator convention. Either is defensible — but worth noting for rigor.

---

Notebook 01 Q&A — adjusted close, correlation drop, VIF alternatives, MSE vs RMSE vs MS(Res):

  1. Adjusted close vs actual close
  Adjusted close retroactively scales historical prices for corporate actions (splits, dividends). For SPY the main effect is dividends — each quarterly payout shifts historical prices so the series reflects total return, not price return. data_loader.py uses yfinance with auto_adjust=True, so "Close" in the parquets is already adjusted. You want this for vol calculations; otherwise a dividend day shows up as a fake ~0.3% "price drop" contaminating log-returns.

  2. Which feature gets dropped in correlation pruning?
  drop_correlated_features (features.py:206-225) uses the upper triangle of the correlation matrix — so the *later* feature in column order is dropped. Column order comes from build_features: log_return → abs_return → oc_return → intraday_range → log_volume → relative_volume → (per window) rolling_mean → rolling_std → rolling_abs_mean.

  From nb 01 output: rolling_abs_mean_{5,10,21} get dropped because they correlate ~0.95+ with rolling_std_{5,10,21}. We keep rolling_std (it comes first in add_rolling_stats). Mechanically correct; incidentally also the more canonical volatility measure. But the algorithm is *purely order-based*, not semantic — reordering add_rolling_stats would flip the choice.

  3. VIF + alternatives
  VIF IS output in nb 01 cell 26 (log_volume=250, intraday_range=127, rolling_std_*≈17-42, etc.). If you're not seeing it, statsmodels isn't installed — it's in requirements.txt but we skipped it in our piecemeal install. `pip install statsmodels` fixes it.

  Nb 01 uses VIF *informationally only*: "no features are dropped on VIF grounds." Modern ML is robust to collinear inputs; VIF matters more for interpretability in linear models than for predictive accuracy.

  Other approaches to the same feature-selection / redundancy question:

  ┌──────────────────────────┬────────────────────────────────────────────┬──────────────────────────────────────────┐
  │ Method                   │ Best for                                   │ Use here?                                │
  ├──────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
  │ AIC / BIC                │ Comparing nested models w/ diff feat count │ Used in HMM model selection (nb 03)      │
  │ Mallows' Cp              │ Subset selection in linear regression      │ Not appropriate (not linear regression)  │
  │ Train/val empirical      │ Any model                                  │ What Optuna effectively does in nb 04/05 │
  │ Correlation threshold    │ Obvious redundancy                         │ We already do this (threshold=0.95)      │
  │ PCA variance ratio       │ Checking effective dimensionality          │ Could add as a diagnostic                │
  │ Condition number of X'X  │ Classical multicollinearity test           │ Similar info to VIF                      │
  └──────────────────────────┴────────────────────────────────────────────┴──────────────────────────────────────────┘

  For the LSTM: best "feature selection" is include everything and let training find what matters. The Phase-6 stationarity fix was a different kind of selection — not redundancy, but distribution quality.

  4. MSE vs RMSE vs MS(Res)
  - MSE = (1/n) Σ(ŷ − y)² — units are squared target (e.g., vol²). Used as the training loss.
  - RMSE = √MSE — same units as target. "Our forecast is off by ~0.004 on average." Used for reporting.
  - MS(Res) = SS(Res) / (n − p) — divides by degrees of freedom (n observations minus p parameters). Classical regression-theory object; gives an unbiased estimator of error variance in linear regression.

  Key difference: MSE divides by n; MS(Res) divides by n − p.

  Does MS(Res) make sense for our LSTMs? No. DOF accounting is fuzzy for neural networks (p often exceeds n; regularization/dropout reduces effective DOF in ways that are hard to quantify). More importantly: we evaluate on a held-out test set, where each observation is independent of training — no n − p correction needed. MSE/RMSE on held-out data is the right choice.

  MS(Res) is only relevant if computing in-sample error on a linear regression and wanting an unbiased variance estimate for F-tests, confidence intervals, etc. — not our context.

---

hmm_model.py verification + report-ready summary:

  Verification: everything checks out ✅

  ┌────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
  │ Section                            │ Check                                                                                                        │ Status │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
  │ _count_gaussian_hmm_params         │ Trans (n·(n−1)) + start (n−1) + means (n·f) + cov (type-dependent)                                           │   ✅   │
  │ _count_gmmhmm_params               │ + mixture weights n·(n_mix−1) + per-mix means and covs                                                       │   ✅   │
  │ _smart_init_hmm                    │ Volatility-quantile label assignment → empirical transition matrix + means + covariances; adds               │   ✅   │
  │                                    │ rng.normal(0, 0.05) offsets for GMMHMM components to break symmetry                                          │        │
  │ _penalised_score                   │ LL − λ · Σ(1 − p_stay_s); returns raw LL if λ=0                                                              │   ✅   │
  │ train_gaussian_hmm / train_gmmhmm  │ Restart 0 = smart init (init_params=""), restarts 1…n use hmmlearn default ("stmc" / "stmcw").               │   ✅   │
  │                                    │ Exception-guarded so one bad restart doesn't kill the sweep.                                                 │        │
  │ ModelComparison                    │ BIC = −2·LL + k·ln(n), AIC = −2·LL + 2k                                                                      │   ✅   │
  │ compare_models                     │ Sorts by val_BIC ascending (lower = better); logs each model                                                 │   ✅   │
  │ viterbi_decode, forward_backward   │ Thin wrappers over model.decode(algorithm="viterbi") and model.predict_proba()                               │   ✅   │
  │ identify_volatile_state            │ Picks state with highest mean abs_return                                                                     │   ✅   │
  │ build_second_order_features        │ hstack([X[lag:], X[:-lag]]) → (T−lag, 2F) — for lag=1 each row = [current, previous]                        │   ✅   │
  │ fit_scaler                         │ StandardScaler, with commented rationale for rejecting RobustScaler                                          │   ✅   │
  └────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

  Two minor observations (not bugs):
  1. identify_volatile_state hardcodes 2 states (calm_state = 1 − volatile_state). Documented in the comment. If you ever try 3+ states, this breaks and needs generalization.
  2. _smart_init_hmm fallback when a quantile group has ≤1 member uses the full X (line 159: g if len(g) > 1 else X for g in groups). For 3000 training rows this never triggers, but it would bias the fallback state toward the global mean. Edge case only.

  No functional bugs. No leakage. Statistically sound.

  ───

  Report-ready summary — HMM component

  Architecture:
  - Model class: GMMHMM (hmmlearn) — first-order Hidden Markov Model with Gaussian-mixture emissions.
  - Hidden states: 2 (interpreted post-hoc as calm and volatile).
  - Emissions per state: mixture of 2 full-covariance Gaussians (n_mix = 2).
  - Parameter count: k = 313 (transition + start + mixture weights + means + covariances across 2 states × 2 mix components × 11 features).
  - Input: 11 features — log_return, abs_return, oc_return, intraday_range, log_volume, rolling_mean/std_{5,10,21} (post correlation-pruning from 14 candidates; rolling_abs_mean_{5,10,21} dropped at correlation threshold 0.95).

  Training procedure:
  - Data: 3,000 training days (2004-01 → 2015-12) after dropping the 20-row rolling-warmup.
  - Standardization: StandardScaler fit on training data only (no val/test leakage); RobustScaler evaluated and rejected because it produced unstable EM convergence.
  - Estimation: Baum-Welch EM, 200 max iterations per fit.
  - Restart protocol: 50 restarts per candidate model. Restart 0 uses volatility-quantile smart init (seeds EM near an economically meaningful calm/volatile split); restarts 1–49 use hmmlearn's default random k-means init.
  - Restart selection: penalised log-likelihood L − λ · Σ_s(1 − p_stay_s) with λ = 3,000 ≈ n_train, following Nystrup et al. (2015–2019). Discourages degenerate rapid-switching local optima that would otherwise win on raw LL.

  Model selection (Phase 2a & 2b):

  Phase 2a — compare GaussianHMM vs GMMHMM(mix=2) vs GMMHMM(mix=3), all 2-state, full covariance:

  ┌──────────────────────────────────┬─────┬────────────┬──────────────┐
  │ Model                            │  k  │  val LL    │  val BIC     │
  ├──────────────────────────────────┼─────┼────────────┼──────────────┤
  │ GMMHMM(states=2, mix=2) ✅       │ 313 │  +9,057.28 │  −15,925.74  │
  │ GMMHMM(states=2, mix=3)          │ 469 │  +9,088.87 │  −14,898.02  │
  │ GaussianHMM(states=2)            │ 157 │  −8,684.22 │  +18,466.33  │
  └──────────────────────────────────┴─────┴────────────┴──────────────┘

  BIC penalizes the extra 156 parameters of mix=3; the tiny LL gain doesn't justify them.

  Phase 2b — 2nd-order Markov via two approaches, both rejected:

  ┌─────────────────────────────────────────────┬──────────┬─────────┬──────────────┬──────────────┐
  │ Approach                                    │ n_states │ n_feats │  val BIC     │ Δ vs winner  │
  ├─────────────────────────────────────────────┼──────────┼─────────┼──────────────┼──────────────┤
  │ 1st-order winner (above)                    │    2     │   11    │  −15,925.74  │      —       │
  │ A: state-space aug (4 composite states)     │    4     │   11    │   +7,936.32  │   +23,862    │
  │ B: feature-space aug (lag-1 concat)         │    2     │   22    │  +23,034.94  │   +38,961    │
  └─────────────────────────────────────────────┴──────────┴─────────┴──────────────┴──────────────┘

  Decision rule: keep 1st-order unless 2nd-order improves BIC by ≥ 10. Neither came close.

  Learned regime structure:
  - Transition matrix: P(stay calm) = 0.9957, P(stay volatile) = 0.9186.
  - Implied average duration: calm ≈ 233 trading days (≈ 11 months); volatile ≈ 12 days (≈ 0.6 months).
  - Regime split (full timeline, 5,564 days): 96.3% calm / 3.7% volatile.
  - State means (scaled, per mixture component) — e.g., volatile state 1 splits into "mild volatile" (abs_return +0.29 std) and "extreme volatile" (abs_return +2.48 std).

  Sanity check — economic validity of the volatile state:

  ┌──────────────────────────┬──────────────────────────────────────────┬────────────────────┬──────────────────────────────────────────────┐
  │ Crisis                   │ Volatile days                            │ Peak P(volatile)   │ Notes                                        │
  ├──────────────────────────┼──────────────────────────────────────────┼────────────────────┼──────────────────────────────────────────────┤
  │ 2008 GFC                 │ 107 (2008-09-16 → 2009-04-14)            │      1.00          │ Onset the trading day after Lehman bankruptcy│
  │ 2020 COVID               │ 47 total; main 40-day run (Feb-28 →      │      1.00          │ Highest confidence of any period             │
  │                          │ Apr-24)                                  │                    │                                              │
  │ 2022 Fed hikes           │ 2 isolated daily flags                   │   0.96 – 1.00      │ Weakly detected — consistent with a moderate │
  │                          │                                          │                    │ bear market, not a liquidity crisis          │
  │ 2016-2019 benign window  │ 0 days                                   │      —             │ Confirms no false positives in quiet periods │
  └──────────────────────────┴──────────────────────────────────────────┴────────────────────┴──────────────────────────────────────────────┘

  Downstream usage:
  - Soft probabilities p_calm, p_volatile (forward-backward) → Phase-4 ensemble weighting: ŷ = p_calm · ŷ_calm + p_volatile · ŷ_volatile.
  - Viterbi hard labels → which LSTM a given day belongs to during regime-specific LSTM training (Phase 3b).
  - Saved artifacts: models/hmm_winner.joblib, models/hmm_scaler.joblib, models/hmm_meta.joblib, data/processed/regime_probabilities.parquet.

  Citations relevant for the methodology section:
  - Rabiner (1989) — original HMM tutorial (EM, Viterbi, forward-backward).
  - Andersen & Bollerslev (1998) — realized volatility formulation for the target.
  - Zhang et al. (2019) — motivates GMMHMM over Gaussian HMM for financial regimes.
  - Nystrup, Hansen & Madsen (2015-2019) — penalized-EM for regime-based models, motivates our λ · Σ(1 − p_stay) term.
  - Lee (2017) — higher-order HMM via state augmentation.

---

Notebook 02 Q&A — return distribution coverage, the three normality tests, volatility clustering + ACF, realized vol distribution, feature scaling:

  1. Why only return distribution? Should we check response vs covariate?

  Actually nb 02 checks all 11 HMM features for normality — section 3 (cells 11-12) loops through every feature in HMM_FEATURES and runs the same tests. All 11 reject normality. Section 1 just spotlights log_return because it's the "lead" variable.

  On "response vs covariate": that instinct comes from regression diagnostics (check residuals vs inputs for linearity/heteroscedasticity). Here the question is different: "within a hypothetical hidden state, how should I model the emission distribution?" → that's a marginal distribution question, which is what normality tests answer. It directly informs the GaussianHMM-vs-GMMHMM choice.

  The "response vs covariate" version DOES show up — subtly — in section 6 (cells 20-23): splits days by median realized vol into pseudo-regimes, then plots each regime's return distribution. That's the closest this notebook gets to "response-conditioned covariate." Levene's test on those two regime-returns gives p=2.8×10⁻⁹³ — massively different variances → regime separation is justified.

  Mild gap: we don't plot log_return vs realized_vol_21d as a scatter, or condition all 11 features on the pseudo-regime. Worth a sentence in methods.

  2. The three normality tests — what does each mean?

  ┌────────────────────────┬───────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │ Test                   │ What it tests                                                 │ When it's most sensitive                             │
  ├────────────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Shapiro-Wilk           │ General departure from normality, based on how sample         │ Small-medium samples; very general                   │
  │                        │ quantiles track a normal's                                    │                                                      │
  │ Jarque-Bera            │ Specifically: skewness = 0 AND kurtosis = 3 (moments match    │ Heavy-tailed deviations from normal                  │
  │                        │ a normal)                                                     │                                                      │
  │ D'Agostino-Pearson     │ Omnibus combining skewness and kurtosis scores into a single  │ Similar to JB but with a different statistic         │
  │                        │ χ² statistic                                                  │ construction                                         │
  └────────────────────────┴───────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  Same null hypothesis (H0: data is normal), different statistics. Running all three is triangulation — if all three reject, the rejection is robust to any one test's idiosyncrasy. Here they all reject massively (p < 10⁻³¹ for every feature), so the conclusion is ironclad.

  "Only a subset of features": cells 11-12 run all 11 HMM features. Cells 6-9 dig deeper into log_return specifically because it's the headline variable and we want to understand how it's non-normal (kurtosis 15, left skew, tail ratio 1.29×) — that shapes the GMMHMM choice.

  3. Volatility clustering, ACF — what are we plotting?

  Volatility clustering: the empirical phenomenon that "large moves tend to be followed by large moves, small by small" — returns aren't iid, their magnitudes persist. It's one of Cont's (2007) eleven stylized facts of financial returns. This is literally the premise of the whole project — if volatility didn't cluster, there'd be no regime structure to detect and the HMM-LSTM architecture would have no reason to work.

  What the plot shows (cell 14, two panels):
  - Left panel — |log_return| over time. Visual eye test. You see obvious "bursts" around 2008, 2020, 2022, where many consecutive days have big absolute moves, separated by long quiet periods.
  - Right panel — ACF (AutoCorrelation Function) of squared returns (r_t²). Each bar at lag k shows Corr(r_t², r_{t-k}²). If returns were iid (no clustering), ACF at all lags > 0 would be ~0. For SPY, ACF stays strongly positive out to 30+ lags.
    - Squared returns (not raw) because Corr(r_t, r_{t-k}) is near-zero (markets are efficient for direction). It's the magnitude that persists.

  The formal test (cell 15) — Ljung-Box on r² at lags 5, 10, 21. Null hypothesis: no autocorrelation. All three p-values are 0.0 → volatility clustering confirmed at overwhelming statistical significance. This is the statistical mandate for the regime-switching model.

  4. What is realized volatility distribution?

  Section 5 characterizes the target variable itself.
  - Raw realized_vol_21d: skewness +3.52, excess kurtosis +17.4. Heavily right-skewed, very heavy-tailed — a few crisis days (2008, 2020) are 5-10× the median vol.
  - log(realized_vol): skewness +0.75, kurtosis +1.07. Near-Gaussian.

  Purpose: justify LOG_TRANSFORM_TARGET = True in the LSTM config. If you trained MSE on the raw target, a few crisis days would dominate the loss gradient. Log-transforming makes the target much more tractable for MSE regression.

  5. Is feature scaling appropriate? Anything missing?

  Yes, appropriate — and section 7 (cell 24) has the right justification. Two concrete reasons scaling is required for HMM:
  1. k-means init uses Euclidean distance; with log_volume ≈ 18 vs log_return ≈ 0.001, clustering is dominated by volume alone and ignores everything else. EM can't recover from this bad init.
  2. Full-covariance matrix conditioning. A 18,000:1 feature range → huge condition number on the 11×11 covariance → near-singular matrices, NaN log-likelihoods.

  Both solved by StandardScaler. RobustScaler was tried (theoretically attractive because IQR < std, preserving crisis signal) but caused degenerate EM convergence — documented in both the notebook and in src/hmm_model.py's fit_scaler. Good empirical choice with the rationale written down.

  Things missing / could strengthen:
  - Scaling is demonstrated on the full df here, but the actual pipeline fits StandardScaler on train only (correct! no leakage) — nb 02 could show a before/after plot using the train-fit scaler applied to val/test, to visually confirm no distribution shift violates the HMM's stationarity assumption.
  - No PCA or intrinsic-dimensionality analysis. With 11 features and high VIF, you could argue the effective dimensionality is ~4-5. Not strictly needed, but would reassure a reviewer worried about parameter count.
  - No outlier handling. We don't winsorize or cap extreme values. This is correct for the HMM (we want crisis outliers — they signal the volatile state) but worth noting explicitly.
  - No explicit stationarity test on the features (e.g., Augmented Dickey-Fuller). Phase 6 later discovered the non-stationarity issue empirically rather than flagging it in EDA. A formal ADF test would have caught it earlier.

---

Notebook 03 — key ideas summary:

  Purpose: train the HMM, pick the best variant, assign regime labels/probabilities to every day in the timeline. Produces artifacts consumed by Phases 3 (regime-specific LSTMs) and 4 (ensemble weighting).

  Structure — three phases:

  ┌───────────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │ Phase     │ Question                             │ Method                                                         │
  ├───────────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ 2a        │ Gaussian or Gaussian-mixture         │ Sweep n_mix ∈ {1, 2, 3}, 2 states, full cov; compare val BIC   │
  │           │ emissions per state?                 │                                                                │
  │ 2b        │ 1st-order or 2nd-order Markov?       │ Two 2nd-order approximations (state-space aug & feature-space  │
  │           │                                      │ aug); retain 1st-order unless BIC improves by ≥ 10             │
  │ 2c        │ Assign regimes + sanity check        │ Viterbi + forward-backward on full timeline; verify volatile   │
  │           │                                      │ state aligns with known crises; save regime_probabilities      │
  └───────────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────────┘

  Key methodological choices:

  1. Selection criterion is validation BIC, not training log-likelihood.
     BIC = −2·LL + k·ln(n) penalizes model complexity. Using val-set BIC (not train) avoids rewarding models that merely overfit training data. Lower BIC = better.

  2. Two-layer defense against degenerate EM solutions.
     EM on HMMs is notorious for converging to "rapid-switching" local optima where states flip every 1-2 days. These can have *higher* training LL than the economically sensible calm/volatile split, so you can't just pick the max-LL restart. Two safeguards:
     (a) Smart init (restart 0): seed EM from volatility-quantile labels instead of hmmlearn's default random k-means — starts EM near an economically meaningful split.
     (b) Penalized restart selection: winner chosen by LL − λ · Σ_s(1 − p_stay_s), with λ ≈ n_train = 3,000. A degenerate model (avg dur 2 days) pays ~3,000 penalty; healthy (avg dur 12-200 days) pays ~250. This lets healthy models win even if degenerate has marginally higher raw LL.

  3. Higher-order Markov tested two ways, both rejected.
     - State-space augmentation: 2² = 4 composite states (s_{t-1}, s_t) → GaussianHMM(n_states=4) on same 11 features. Val BIC +23,862 *worse*.
     - Feature-space augmentation: append lag-1 features → 22-dim observations, 2 states. Val BIC +38,961 *worse*.
     Conclusion: 1st-order 2-state is the right structure for equity regimes. Second-order memory adds parameter burden without commensurate fit improvement.

  4. Viterbi vs forward-backward — both saved, different uses.
     - Viterbi → hard labels (one state per day). Used in Phase 3b to partition training data per regime.
     - Forward-backward → soft probabilities p_calm, p_volatile summing to 1. Used in Phase 4 ensemble weighting ŷ = p_calm·ŷ_calm + p_volatile·ŷ_volatile.

  5. Economic sanity check, not just statistical fit.
     After decoding, verify the "volatile" state aligns with known crises — 2008 GFC, 2020 COVID, 2022 Fed hikes. If the model picks up noise regimes that don't correspond to real events, BIC doesn't save you. In our case: 107 volatile days capture GFC (onset day after Lehman), 47 capture COVID (P≈1.0 throughout), 2 weakly flag 2022 bear market. 0 false positives in the benign 2016-2019 window. Passes.

  6. Full-timeline decoding across splits.
     Viterbi is run on np.vstack([train_X, val_X, test_X]) as one continuous sequence so transitions are preserved across split boundaries. Alternative (decode each split independently) would lose context at boundaries. This doesn't leak data — the trained model is fixed; Viterbi only uses the current sequence for inference.

  Winning model summary:

  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │ Field                │ Value                                                                    │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Model class          │ GMMHMM (first-order)                                                     │
  │ Hidden states        │ 2                                                                        │
  │ Emissions per state  │ 2 full-covariance Gaussians                                              │
  │ Parameters (k)       │ 313                                                                      │
  │ Features (11)        │ log_return, abs_return, oc_return, intraday_range, log_volume,           │
  │                      │ rolling_mean/std_{5,10,21}                                               │
  │ Val log-likelihood   │ +9,057.28                                                                │
  │ Val BIC              │ −15,925.74                                                               │
  │ P(stay calm)         │ 0.9957 → avg calm duration ≈ 233 days (11 months)                        │
  │ P(stay volatile)     │ 0.9186 → avg volatile duration ≈ 12 days (0.6 months)                    │
  │ Regime split         │ 96.3% calm / 3.7% volatile (over 5,564 days)                             │
  └──────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  Outputs:
  - models/hmm_winner.joblib — fitted GMMHMM
  - models/hmm_scaler.joblib — StandardScaler fit on train only
  - models/hmm_meta.joblib — dict with winning_label, BIC/LL, volatile_state ID, feature list
  - data/processed/regime_probabilities.parquet — p_calm, p_volatile, viterbi_state, regime_label indexed by Date

  What's novel vs "use hmmlearn out-of-the-box":
  - Volatility-quantile smart init
  - Penalized log-likelihood for restart selection (λ=3000)
  - Explicit 1st vs 2nd order Markov comparison via two different augmentation strategies
  - Economic sanity check with crisis alignment — not just BIC minimization

---

Why is the regime-split idea failing? And why do baseline predictions appear "shifted"?

  Preliminary hypotheses — why regime-split is failing:

  1. Volatile LSTM is starved for data. 181 training windows is an order of magnitude too few to train a real LSTM — it probably collapses to a near-constant output (~0.0175, matching our observation in the predictions).
  2. Domain shift at inference. Both regime LSTMs are applied to every test window, not just their own regime. So the volatile LSTM is being asked to predict on calm days 97% of the time — way outside its training distribution.
  3. Calm LSTM loses the transition signal. Excluding volatile days from training also excludes the most informative periods — the days that precede regime shifts. So the calm LSTM is trained on the least-predictive subset of the data.
  4. HMM is reactive, not predictive. P(volatile) only spikes after volatile returns enter the observation window. The ensemble gate can't shift to the volatile LSTM before the transition; by the time it does, vol is already high.
  5. Noisy partition at boundaries. Windows straddling regime boundaries get assigned by majority vote of Viterbi states — creating mislabeled training data near exactly the transition periods we care most about.

  On the "shift" in baseline predictions:

  This is the classic finding in vol forecasting — the LSTM converges to a persistence predictor. Since realized vol is highly autocorrelated (that's the whole volatility-clustering story), "predict next-21-day vol ≈ recent-21-day vol" is a very strong local optimum. The shift you're seeing is just the model tracking recent vol rather than anticipating future vol. Transitions (COVID, GFC) are largely exogenous — they don't leave a forecastable signal in past prices, so even a perfect model can only react to them.

  ───

  Verified with data — these hypotheses are damning evidence:

  Finding 1: The Volatile LSTM is literally constant.

  Prediction statistics across the test set:

  ┌──────────┬──────────┬─────────────────┐
  │  Model   │ Pred std │   Pred range    │
  ├──────────┼──────────┼─────────────────┤
  │ Target   │ 0.0044   │ —               │
  │ Naive    │ 0.0046   │ 0.003 → 0.033   │
  │ Baseline │ 0.0039   │ 0.007 → 0.031   │
  │ Calm     │ 0.0069   │ 0.007 → 0.061   │
  │ Volatile │ 0.0001   │ 0.0175 → 0.0183 │
  │ Ensemble │ 0.0057   │ 0.007 → 0.049   │
  └──────────┴──────────┴─────────────────┘

  The volatile LSTM's predictions vary by less than 0.8 basis points across the entire test set. It outputs ~0.0179 regardless of input — essentially the mean of its 181 training windows' targets. It's not a model, it's a constant. With only 181 training windows against an LSTM of hidden_size=64, n_layers=3, dropout=0.42 (~17k+ parameters), Optuna converged to whatever minimized MSE on the tiny training set — which turned out to be "output the mean." When p_volatile > 0 in the ensemble, this constant pulls predictions up toward 0.018, way above typical calm-day vol (~0.008).

  Finding 2: The shift is real and measurable — all our LSTMs are persistence predictors.

  Cross-correlation peaks at negative lags for every non-constant model (negative lag = prediction is shifted later than the target, i.e., reactive):

  ┌────────────────────────┬───────────────┬──────────┬───────────────┐
  │         Model          │ Zero-lag corr │ Best lag │ Best-lag corr │
  ├────────────────────────┼───────────────┼──────────┼───────────────┤
  │ Naive (rolling_std_21) │ +0.46         │ −21 days │ +1.00         │
  │ Baseline LSTM          │ +0.56         │ −16 days │ +0.83         │
  │ Calm LSTM              │ +0.46         │ −18 days │ +0.82         │
  │ Ensemble               │ +0.53         │ −17 days │ +0.70         │
  └────────────────────────┴───────────────┴──────────┴───────────────┘

  The naive baseline correlates PERFECTLY with target at lag −21. This is mathematically guaranteed: rolling_std_21 at time t and realized_vol_21d at time t−21 are computed from the same 21 returns (just re-indexed by which end of the window we're labeling from). The naive is literally the target shifted back by 21 days.

  Our LSTMs are almost as good: they achieve correlation 0.82-0.83 at lag −16 to −18. In other words, the LSTMs are slightly-worse persistence predictors with a ~17-day shift — marginally better than pure rolling_std only in the sense that they smooth the persistence signal.

  Why this is happening — diagnosis:

  The combination of (a) 21-day-forward target, (b) no features containing forward-looking information, and (c) highly persistent vol, forces any gradient-trained model toward one solution: output something close to the recent past's vol. The model's effective function is ŷ(t) ≈ f(rolling_vol_at_t) where f is almost the identity. Adding more parameters doesn't help — there's no additional signal to extract from the input features about future vol beyond what recent vol already tells you.

  What this means for the regime-split story:

  1. Regime split doesn't help because the volatile LSTM can't learn anything from 181 windows — it's starved, degenerate, and then applied to the 97% of test days where it has no relevance.
  2. Calm LSTM is trained on the less-informative subset. By excluding the 4.5% of volatile training days, we threw away the transition periods where vol dynamics are most variable (and hence most informative for learning anything beyond pure persistence).
  3. HMM is also reactive. P(volatile) only rises after volatile returns enter the observation window. The ensemble's gate can't proactively shift toward the volatile model — it responds to vol shocks with the same lag as the LSTMs.

  So our ensemble is "two reactive predictors gated by a reactive detector." It's structurally incapable of beating persistence on a 21-day-forward task with only price/volume features.

  Honest framing for the paper:

  This is a legitimate negative result worth writing up clearly:

  - The regime-switching hypothesis (HMM-LSTM > plain LSTM) requires the regime signal to be informative about future vol beyond what recent vol already tells you. On SPY 2004-2026 with price/volume features only, the regime information at time t is almost fully redundant with recent-magnitude information that the baseline LSTM already has access to through its input window.
  - Volatility prediction on 21-day forward horizons is fundamentally close to persistence. Our deep model captures ~70% of the variance the naive captures — slightly worse, not better.
  - Where an ensemble would plausibly help is on volatile-dominant days, where the point-estimate MSE gap (5e-5 vs 8e-5) is directionally correct. But the underpowered n=27 subset prevents a significance claim.

  Concrete next steps (if there's time):

  1. Activate sentiment + put/call features (the proposal's Ho et al. / Gupta et al. references). These contain signal that isn't in price history — this is the single highest-leverage change you could make.
  2. Fix the volatile LSTM's data starvation — either (a) extend training to pre-2004 for more crisis data, (b) augment with a simpler "always output rolling_std_21" baseline for the volatile regime instead of a neural net.
  3. Clip volatile predictions when p_volatile < threshold — simple fix for the constant-output contamination.
  4. Multi-seed robustness study (documented as open item #4 in discussion.md) — single-run results are not reliable here given the training stochasticity we already observed.
  5. Gating network instead of HMM soft probabilities — the proposal's "considered but out of scope" idea. A learned gate trained on (recent features → which-model-to-trust) could potentially learn forward-looking gating behavior.

---

Phase 1 gating experiment (linear stacking) — results and why it fails:

  Setup: Ridge/OLS stacking meta-learner taking as inputs [baseline_pred, calm_pred, volatile_pred, p_calm, p_volatile], fit on train to predict realized_vol_21d. Alpha swept on val; best model evaluated on test. Script: src/gate_stacking.py.

  Data:
  - Train rows: 2,960 | Val rows: 1,048 | Test rows: 1,434

  Alpha sweep (val MSE):
  - OLS (α=0):       6.68e-5  (exploded — p_calm+p_volatile=1 collinearity)
  - Ridge α=0.01:    4.34e-5
  - Ridge α=0.1:     4.19e-5  ← best
  - Ridge α=1:       4.57e-5
  - Ridge α=10:      4.76e-5
  - Ridge α=100:     5.01e-5

  Learned coefficients (Ridge α=0.1):
  - intercept:   +7.74e-3
  - baseline:    +0.167
  - calm:        +0.361
  - volatile:    −0.002   ← ~zero; gate learned to ignore volatile LSTM
  - p_calm:      −0.004   ← ~zero; HMM probs add no linear signal
  - p_volatile:  +0.004   ← ~zero; same

  Test-set performance:

  ┌──────────────┬──────────────┬────────┐
  │ Model        │ MSE          │ MAPE   │
  ├──────────────┼──────────────┼────────┤
  │ Naive        │ 2.20e-5      │ 34.6%  │
  │ Baseline     │ 1.59e-5      │ 25.1%  │ ← best
  │ HMM-Ensemble │ 2.61e-5      │ 31.9%  │
  │ Gated        │ 1.87e-5      │ 28.1%  │
  └──────────────┴──────────────┴────────┘

  Diebold-Mariano tests (gated vs each baseline):

  ┌────────────────────────────┬──────┬───────┬───────────────────────────────┐
  │ Comparison                 │ Loss │ p     │ Verdict (α=.05)               │
  ├────────────────────────────┼──────┼───────┼───────────────────────────────┤
  │ Gated vs Naive             │ MAE  │ 0.002 │ Gated wins **                 │
  │ Gated vs Baseline          │ MSE  │ 0.43  │ tie (point estimate worse)    │
  │ Gated vs Baseline          │ MAE  │ 0.57  │ tie                           │
  │ Gated vs HMM-Ensemble      │ MSE  │ 0.31  │ tie (directionally better)    │
  │ Gated vs HMM-Ensemble      │ MAE  │ 0.16  │ tie (directionally better)    │
  └────────────────────────────┴──────┴───────┴───────────────────────────────┘

  Summary: Gated beats HMM-ensemble directionally but ties with the plain baseline LSTM on both MSE and MAE. Gated does NOT add value over baseline.

  Why the gate fails — structural diagnosis:

  Two independent conditions combine to make any gating mechanism degenerate on this data:

  1. The gating signal barely varies. p_volatile > 0.5 on only 207 / 5,564 days (3.7%) full timeline, and only 27 / 1,434 test days. For 97% of days the HMM is essentially certain it's calm, so the weight allocation is approximately fixed at (1.0, 0.0). There is no useful gating decision to make most of the time.

  2. The gate's "volatile expert" is a constant. Even on the 3% of days where the gate would want to activate the Volatile LSTM, that model outputs ~0.0179 regardless of input (finding F2 from the report — Volatile LSTM pred std = 0.0001). So "gating to the volatile expert" gives you exactly one more signal: a biased constant that's too high on most days. Not information — just an offset.

  Combined: the gate's entire job reduces to "should I blend in a constant 0.018?" and the answer is almost always no. Ridge correctly zeroed out the `volatile` coefficient and the HMM-prob coefficients, leaving only `baseline` and `calm` as inputs — but those are near-duplicates of each other (both shifted persistence predictors), so the gate couldn't combine them better than baseline alone.

  Implication for Phase 2 (MLP gate): killed. Phase 2 faces the same two structural problems a nonlinear architecture can't solve:
  - If the gating signal is nearly constant 97% of the time, there's nothing to gate *on*.
  - If the thing you're gating *toward* is a biased constant, there's nothing useful to gate *to*.
  Linear Ridge already zeroed out those inputs; a nonlinear gate has no additional structure to discover. Decision: skip Phase 2. Write up Phase 1 as the definitive answer.

  What would actually unblock this (if there's time):
  - Activate the sentiment + put/call features. This is the only change with the potential to introduce a predictive signal not already in recent-vol. The proposal's Ho et al. (2012) and Gupta et al. (2023) motivations are specifically about features that DON'T reduce to persistence.
  - Fix the volatile LSTM's data starvation (pre-2004 data, or replace its neural net with a simpler non-degenerate predictor). Even then, gating only helps if the volatile model actually varies meaningfully across volatile days.
  - Extend the test window pre-2020 to include GFC — more volatile days give us statistical power to test regime-specific claims.

---

Why only n=181 volatile training windows over 10+ years? And is daily frequency enough?

  Yes, 181 is reasonable given the data — here is the audit.

  Volatile days per split (HMM Viterbi on full timeline):

  ┌───────────────────────────┬─────────────┬───────────────┬───────┐
  │ Split                     │ Total days  │ Volatile days │  %    │
  ├───────────────────────────┼─────────────┼───────────────┼───────┤
  │ Train (2004-2015)         │ 3,000       │ 134           │ 4.5%  │
  │ Val   (2016-01 → 2020-04) │ 1,089       │ 41            │ 3.8%  │
  │ Test  (2020-05 → 2026-03) │ 1,475       │ 32            │ 2.2%  │
  └───────────────────────────┴─────────────┴───────────────┴───────┘

  Where are those 134 training-volatile days concentrated? Almost entirely in the 2008 GFC (107 days, 2008-09-16 → 2009-04-14). The remaining ~27 days are scattered in short clusters (August 2011 debt ceiling, flash crashes, etc.) — most of them too short to produce majority-volatile 42-day windows.

  Window math at seq_len=42, majority threshold 22:
  - A contiguous 107-day GFC run produces ~108 majority-volatile windows (windows shifted by up to 20 days outside the run still capture 22+ volatile days).
  - Scattered short volatile clusters elsewhere contribute ~70 more.
  - Total: ~181 windows. Matches.

  Crucially: COVID doesn't help training. COVID's main vol period (2020-02-28 → 2020-04-24, 40 days) falls in val, not train. So of "2008 + COVID" — the two crises you'd hope to train on — only GFC is available to the training set. The volatile LSTM is effectively trained on ONE crisis.

  Independent-samples count — even worse than 181

  Those 181 windows are heavily overlapping (each shares 41 of 42 days with its neighbour). If we care about statistical independence:
  - 134 volatile days / 42-day window ≈ 4-5 independent volatile events
  - Essentially: GFC 2008 (dominating) + 4-ish scattered short episodes

  For a 17k-parameter LSTM, you'd want orders of magnitude more diverse examples. This is why the volatile LSTM collapses to a constant — it effectively memorised "GFC-like days have vol ≈ 0.018."

  Is daily frequency enough for a 21-day-forward target?

  No, arguably not. The target is a monthly-scale prediction. Non-overlapping 21-day windows give:
  - Training: 3,000 days / 21 ≈ 143 independent monthly observations over 12 years
  - Of those, volatile-dominant: ~7-14 "independent volatile months"

  The 3,000 sliding windows share 20/21 of their inputs with their neighbours — they are not 3,000 independent samples. For an LSTM learning a 21-day-forward mapping, the effective sample size is closer to 143 than 3,000. For the volatile subset, it is single-digit.

  Three axes that could address this (ordered by leverage)

  ┌───────────────────────────────────┬──────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
  │ Approach                          │ What it does                                             │ Downside                                             │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
  │ Higher-frequency data             │ 78× more observations per trading day (5-min bars).      │ Different problem — target changes, microstructure   │
  │ (e.g., 5-minute bars)             │ Volatile "days" become volatile 5-minute windows.        │ noise dominates at HF. Would need different model    │
  │                                   │ Much richer training set for the volatile regime.        │ formulation.                                         │
  │                                   │                                                          │                                                      │
  │ Extend history                    │ Picks up 2001 dot-com, LTCM 1998, Asian crisis 1997.     │ Diminishing returns — volatile base rate (~4%) is    │
  │ (pre-2004 SPY back to ~1994)      │ ~2× the training range.                                  │ structural, not an artifact. 20 years → maybe 300    │
  │                                   │                                                          │ volatile days, still not a lot.                      │
  │                                   │                                                          │                                                      │
  │ Transfer learning / asset pooling │ Train the volatile LSTM on volatile periods across       │ Regimes are correlated across US equities, so "more  │
  │                                   │ multiple assets (SPY, QQQ, IWM, international indices).  │ assets" ≈ mild augmentation, not true multiplication.│
  │                                   │ Orders-of-magnitude more data in principle.              │ More effective for global crises than scattered      │
  │                                   │                                                          │ shocks.                                              │
  └───────────────────────────────────┴──────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

  Sharper paper framing — upgrade from "our regime split didn't work" to:

  The regime-switching framework with per-regime deep models faces a structural data-scarcity problem on daily-frequency equity data: volatile regimes are rare (~3.7% base rate for SPY), concentrated in individual crises (GFC dominates our training window), and — given a 21-day-forward prediction target — the effective number of independent volatile training events is in the single digits. This is not a bug in our implementation; it is a fundamental limit of applying deep learning to rare regimes in daily-frequency financial data.

  This elevates the finding from "we tried and it didn't help" to "here is a principled reason this class of approach is hard on data like ours — and here are the three axes along which the problem could be made tractable." Much better paper material.

---

How to actually get meaningfully more volatile-regime training data (beyond "just pool more assets"):

  Key insight — most naive pooling fails: Because SPY is an ETF holding the largest US equities, and because systemic crises (2008 GFC, 2020 COVID) hit virtually every risk asset simultaneously, pooling SPY with QQQ + IWM + most individual US stocks just *duplicates* the same volatile events. The win in data augmentation comes specifically from *non-systemic* crises where one market spikes but SPY stays quiet — i.e., crises SPY did NOT feel at the same intensity.

  Taxonomy of "independence from SPY's volatile regimes" (Levels A through D, decreasing in SPY similarity):

  Level A — Geographic independence (regional equity indices, same asset class):

  ┌──────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐
  │ Asset                        │ Regional-only vol events SPY barely felt                                     │
  ├──────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Euro STOXX 50 / EZU          │ Euro sovereign debt crisis 2010-12; Brexit 2016; 2022 energy crisis          │
  │ Nikkei 225 / EWJ             │ 2011 Tōhoku earthquake + Fukushima; BOJ yield-curve-control events           │
  │ Hang Seng / FXI              │ 2015 CNY devaluation + mainland crash; 2018 trade war; property bust 2021-22 │
  │ EM aggregate (EEM / VWO)     │ 2013 taper tantrum; 2014-16 Russia/Brazil; Turkey 2018                       │
  └──────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘

  Pros: same asset class → target definition + dynamics trivially transfer; yfinance-available; many regional crises SPY didn't feel.
  Cons: correlation with SPY is still ~0.4-0.6 normally and ~0.8+ during global crises — not pure independence.

  Level B — Asset-class independence (genuinely different risk factors):

  ┌────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │ Asset                  │ What it teaches the model                                                   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ TLT (long Treasuries)  │ Rate-driven vol: 1994 bond crash, 2013 taper tantrum, 2022 inflation spike, │
  │                        │ UK gilt crisis 2022                                                         │
  │ HYG (high-yield)       │ Credit-spread vol: 2008, 2015-16 energy HY bust, 2020 briefly              │
  │ GLD (gold)             │ Safe-haven vol: 2011-13 boom/bust, geopolitical events                      │
  │ USO (oil)              │ Commodity supply shocks: 2014-16 collapse, 2020 negative pricing, 2022 war  │
  │ DXY (dollar index)     │ FX vol: strong-dollar regimes, emerging-market crises                       │
  └────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  Pros: genuinely orthogonal risk factors; crises in these assets rarely touch SPY; many more independent volatile episodes available.
  Cons: vol dynamics may genuinely differ by asset class — a gold LSTM may not transfer to SPY. Transfer-learning assumption would need validation via ablation.

  Level C — Sectoral "independence" within equities (partial, since sectors are inside SPY):

  - XLE 2014-16: oil crash tanked energy while SPY went mildly up (~2 years of energy-specific high-vol training).
  - XLF 2008: banking crisis more severe in XLF than SPY aggregate.
  - XLK 2022: rate-hike cycle crushed tech while broader market held up.
  - Pros: easy; same dynamics.
  - Cons: only partially independent — most sector vol correlates tightly with market vol. Small marginal gain.

  Level D — Synthetic augmentation (bypasses the data problem entirely):

  - GARCH-simulated vol regimes: fit GARCH(1,1) or regime-switching GARCH to SPY, then simulate long synthetic histories. Train on real + synthetic.
  - Block bootstrap from existing volatile periods: sample with replacement from GFC days to create "more GFC-like" training examples.
  - Noise augmentation on existing volatile windows: add controlled Gaussian noise to recreate near-duplicates.
  - Pros: unlimited training data; regime base rate is tunable.
  - Cons: the model learns synthetic statistics, not market dynamics. If GARCH's model is wrong, the LSTM inherits that bias. A separate project really.

  Ranking for this project (what to try if we had more time):

  1. EM + European index pooling (Level A) — same asset class, different crises, minimal methodological handwaving. Best shot at "more data without changing the problem."
  2. Multi-asset transfer with rates/credit (Level B — TLT, HYG) — higher-risk methodologically but potentially higher reward. Needs explicit ablations ("does a vol model trained on bonds transfer to equities?").
  3. Sector ETFs (Level C) — cheap, small gain; good supplementary experiment.
  4. Synthetic augmentation (Level D) — interesting but becomes a separate project; requires its own validation.

  The deeper open question — is there a universal "volatile regime"?

  Transfer learning only works if volatile-regime dynamics are *shared* across asset classes. If GFC, Euro crisis, oil crash, and currency collapses are all fundamentally the same phenomenon (heavy-tailed returns + momentum + clustering), transfer is justified. If each crisis has its own qualitatively distinct dynamics, pooling is just averaging noise — and the data-scarcity problem genuinely can't be solved by pooling.

  Mini-experiment that would answer this without retraining: compute autocorrelation + tail-index profile of volatile days in each asset class and measure similarity. If similar → transfer-learning path is defensible. If wildly different → another research finding: "volatile-regime dynamics are asset-specific; pooling does not solve data scarcity."

  This is itself a publishable additional contribution: "we characterize the transferability of volatile-regime dynamics across asset classes and find [X]." If we extended the project, that's a defensible direction independent of whether it improves forecasting.

---

Beyond put/call ratio — richer options-derived forward-looking features:

  Key insight: put/call volume ratio is the blunt instrument of options-derived features. It counts contracts but ignores (a) where on the strike chain the activity is, (b) what prices people are paying, (c) how asymmetric the risk pricing is across strikes. Options markets encode much richer information.

  Taxonomy, simplest → most informative:

  Tier 1 — Volume-based (what the proposal/partner currently targets):
  - Put/call volume ratio: cheap, noisy; treats a $1 OTM put the same as a $50 ATM put.
  - Put/call open interest ratio: positioning rather than flow; smoother but laggier.

  Tier 2 — Moneyness-weighted (filters for meaningful bets):
  - OTM put volume specifically (pure crash hedging).
  - OTM call volume specifically (speculation for big up-moves).
  - Delta-weighted ratios: weight each contract by how sensitive it is to the underlying.

  Tier 3 — Price / IV-derived (where it gets interesting):
  - IV skew = IV(25-delta put) − IV(25-delta call). High positive = puts expensive relative to calls = market pricing crash risk asymmetrically. Crash-fear indicator that pure put/call ratio misses.
  - Risk reversal = IV(OTM call) − IV(OTM put). Negative = crash fear; positive = rally chasing.
  - Butterfly premium = cost of wings − body of a butterfly spread. High = market pricing fat-tailed moves in either direction.
  - Variance risk premium = implied vol − realized vol. Divergence = market anticipating regime change.

  Tier 4 — Term-structure / official CBOE indices (free, daily, easy — the canonical feature set):

  ┌──────────────┬───────────┬─────────────────────────────────────────────────────────────────────────┐
  │ Feature      │ Ticker    │ What it tells you                                                       │
  ├──────────────┼───────────┼─────────────────────────────────────────────────────────────────────────┤
  │ VIX          │ ^VIX      │ 30-day IV — market's forward vol forecast                               │
  │ VIX3M        │ ^VIX3M    │ 3-month IV — longer-horizon vol                                         │
  │ VIX9D        │ ^VIX9D    │ 9-day IV — very near-term                                               │
  │ SKEW index   │ ^SKEW     │ Price of S&P 500 left tail (crash risk premium). 100 = Normal;          │
  │              │           │ 130+ = significantly elevated tail pricing                              │
  │ VVIX         │ ^VVIX     │ Vol-of-vol — uncertainty about future vol levels                        │
  └──────────────┴───────────┴─────────────────────────────────────────────────────────────────────────┘

  All on yfinance, zero custom options-data engineering.

  Tier 5 — Dealer positioning (specialized paid data; out of scope for our project):
  - GEX (dealer gamma exposure): how much dealers need to hedge. Negative GEX = hedging amplifies moves (vol-spiky regime).
  - Vanna / charm: higher-order dealer sensitivities; drives price around OpEx.
  - Requires SpotGamma / Squeezemetrics / similar paid feeds.

  Concrete recommendations for the paper (ordered by leverage):

  1. VIX + VIX3M + their ratio: single biggest win. The term-structure ratio VIX/VIX3M is itself a stress signal (> 1 = backwardation = market expecting near-term stress).
  2. SKEW index: directly measures the crash-risk premium that put/call ratio is trying to proxy — but much cleaner because it's *price*-based not *volume*-based. Genuinely more informative than put/call ratio.
  3. VVIX: elevated VVIX precedes vol regime changes — a forward-looking warning signal.
  4. IV skew / risk reversal: require SPX option chain data. Cleaner than volume ratios but more engineering cost.
  5. Full strike-weighted option-chain features: the honest version of "what the put/call ratio is trying to measure." Only with paid/historical chain data.

  Honest caveat on put/call ratio alone:

  If your partner is adding put/call ratio, they should grab SKEW and VVIX at the same time. These are more informative, from the same market (options on S&P 500), with essentially identical data-engineering cost. Using put/call ratio in isolation when SKEW and VVIX are free is leaving signal on the table.

  How this ties back to the structural-limits framing:

  All Tier 3/4 features are forward-looking by construction — they're derived from prices of contracts on future outcomes. They're exactly what F1/F4 say we need to escape the persistence shift. Future-work framing:

  "Options markets encode richer forward-looking information than put/call volume ratio alone. Specifically, the SKEW index (CBOE) measures the market-implied left-tail premium, and VIX term-structure slope captures anticipated changes in vol regimes. These features — which we explicitly excluded to characterize the structural limits of price-only models — are the most direct route to a genuinely predictive (as opposed to reactive) vol forecaster."


  