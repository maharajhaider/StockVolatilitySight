"""
HMM regime detection module.

Phase 2a — Model comparison
-----------------------------
Trains GaussianHMM and GMMHMM with varying n_mix values on the training set
and compares them using:

  - Marginal log-likelihood  P(O | model)  via the forward algorithm
    → hmmlearn exposes this as model.score(X, lengths)
    → No labels are needed: the hidden states are marginalised out by the
       forward algorithm. score() sums log P(o_t | o_{1:t-1}, model) over
       all t, integrating over all possible state sequences.

  - BIC  = -2 * log_likelihood + k * log(n)
  - AIC  = -2 * log_likelihood + 2 * k

  where k = number of free parameters and n = number of observations.

Phase 2b — Higher-order Markov
--------------------------------
hmmlearn only supports first-order HMMs natively.  A 2nd-order Markov chain
is approximated by augmenting the state space:  with S base states, a 2nd-order
model is represented as a 1st-order model over S² composite states (s_{t-1}, s_t).
We train GaussianHMM / GMMHMM with n_components = S² and compare BIC against
the first-order model.

Phase 2c — Viterbi decoding and sanity check
---------------------------------------------
Given the winning model, decode the most likely state sequence and plot it
against SPY price to verify the "volatile" state captures known crises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


# ── Parameter counting ────────────────────────────────────────────────────────

def _count_gaussian_hmm_params(n_states: int, n_features: int, cov_type: str) -> int:
    """
    Free parameters in a GaussianHMM (first-order, full emission).

    Components
    ----------
    - Transition matrix    : n_states * (n_states - 1)   (each row sums to 1)
    - Start probabilities  : n_states - 1
    - Means                : n_states * n_features
    - Covariances          :
        'full'    → n_states * n_features * (n_features + 1) / 2
        'diag'    → n_states * n_features
        'spherical'→ n_states
        'tied'    → n_features * (n_features + 1) / 2
    """
    trans = n_states * (n_states - 1)
    start = n_states - 1
    means = n_states * n_features

    if cov_type == "full":
        cov = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        cov = n_states * n_features
    elif cov_type == "spherical":
        cov = n_states
    elif cov_type == "tied":
        cov = n_features * (n_features + 1) // 2
    else:
        cov = n_states * n_features  # fallback

    return trans + start + means + cov


def _count_gmmhmm_params(n_states: int, n_mix: int, n_features: int, cov_type: str) -> int:
    """
    Free parameters in a GMMHMM.

    Each state has n_mix Gaussian components plus mixture weights.
    Mixture weights: n_states * (n_mix - 1)
    Means: n_states * n_mix * n_features
    Covariances: n_states * n_mix * <per-Gaussian cov params>
    Plus transition + start as above.
    """
    trans = n_states * (n_states - 1)
    start = n_states - 1
    mix_weights = n_states * (n_mix - 1)
    means = n_states * n_mix * n_features

    if cov_type == "full":
        cov = n_states * n_mix * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        cov = n_states * n_mix * n_features
    elif cov_type == "spherical":
        cov = n_states * n_mix
    else:
        cov = n_states * n_mix * n_features

    return trans + start + mix_weights + means + cov


# ── Smart (volatility-quantile) initialisation ───────────────────────────────

def _smart_init_hmm(
    model: hmm.GaussianHMM | hmm.GMMHMM,
    X: np.ndarray,
    vol_feature_idx: int = 1,
    rng_seed: int = 0,
) -> None:
    """
    Warm-start HMM parameters from volatility-quantile state assignments.

    Splits observations into n_components groups ordered by a volatility-proxy
    feature (default index 1 = abs_return in HMM_FEATURES), then sets:

      • transmat_   — estimated from consecutive label transitions
      • startprob_  — uniform
      • means_      — empirical group means
      • covars_     — regularised empirical group covariances
      • weights_    — uniform (GMMHMM only)

    For GMMHMM, all n_mix mixture components within a state are initialised
    near the group mean with small random offsets so EM can differentiate them.

    Caller must set model.init_params = "" before calling fit() to prevent
    hmmlearn from overwriting this initialisation with its own k-means step.

    Why this matters
    ----------------
    hmmlearn's default k-means initialisation clusters in feature space.
    With 11 correlated rolling statistics, k-means groups observations by
    their absolute level (e.g. all high-volume days) regardless of economic
    regime.  The resulting EM run converges to degenerate rapid-switching
    solutions where both states alternate every 1–2 days.

    By seeding from volatility quantiles we start EM near the economically
    meaningful solution (calm / volatile regime split), dramatically
    increasing the probability of convergence to a useful local optimum.
    """
    n_obs, n_features = X.shape
    n_states = model.n_components
    rng = np.random.default_rng(rng_seed)

    # ── 1. Volatility-based label assignment ─────────────────────────────────
    vol_signal = np.abs(X[:, vol_feature_idx])
    thresholds = np.percentile(vol_signal, np.linspace(0, 100, n_states + 1)[1:-1])
    labels = np.clip(np.searchsorted(thresholds, vol_signal), 0, n_states - 1)

    groups = [X[labels == s] for s in range(n_states)]
    groups = [g if len(g) > 1 else X for g in groups]   # fallback if empty

    # ── 2. Transition matrix from label sequence ─────────────────────────────
    transmat = np.full((n_states, n_states), 1e-4)
    for t in range(n_obs - 1):
        transmat[labels[t], labels[t + 1]] += 1
    model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)

    # ── 3. Uniform start probabilities ───────────────────────────────────────
    model.startprob_ = np.full(n_states, 1.0 / n_states)

    # ── 4. Emission parameters ────────────────────────────────────────────────
    state_means = np.array([g.mean(axis=0) for g in groups])
    state_covars = np.array([
        np.cov(g.T) + 1e-4 * np.eye(n_features) for g in groups
    ])

    if isinstance(model, hmm.GaussianHMM):
        model.means_ = state_means
        if model.covariance_type == "full":
            model.covars_ = state_covars
        elif model.covariance_type == "diag":
            model.covars_ = np.array([np.diag(c) for c in state_covars])
        elif model.covariance_type == "tied":
            model.covars_ = state_covars.mean(axis=0)
        elif model.covariance_type == "spherical":
            model.covars_ = np.array([np.diag(c).mean() for c in state_covars])

    elif isinstance(model, hmm.GMMHMM):
        n_mix = model.n_mix
        model.weights_ = np.full((n_states, n_mix), 1.0 / n_mix)
        # Small per-component offsets so EM can differentiate mixture components
        model.means_ = np.array([
            [state_means[s] + rng.normal(0, 0.05, n_features) for _ in range(n_mix)]
            for s in range(n_states)
        ])
        if model.covariance_type == "full":
            model.covars_ = np.array([
                [state_covars[s] for _ in range(n_mix)]
                for s in range(n_states)
            ])
        elif model.covariance_type == "diag":
            diag_covars = np.array([np.diag(c) for c in state_covars])
            model.covars_ = np.array([
                [diag_covars[s] for _ in range(n_mix)]
                for s in range(n_states)
            ])


# ── Penalised score ───────────────────────────────────────────────────────────

def _penalised_score(model: hmm.GaussianHMM | hmm.GMMHMM, X: np.ndarray, lam: float) -> float:
    """
    Penalised log-likelihood for HMM model selection.

    penalised_LL = log P(O | model) − λ · Σ_s (1 / avg_duration_s)

    Since avg_duration_s = 1 / (1 − transmat_[s,s]), this simplifies to:

        penalised_LL = log P(O | model) − λ · Σ_s (1 − transmat_[s,s])

    A degenerate model (both states switch every ~2 days) has
        Σ_s(1 − p_stay) ≈ 1.0
    A model with economically meaningful regimes (20–200 day durations) has
        Σ_s(1 − p_stay) ≈ 0.01–0.10

    The penalty weight λ ≈ n_observations makes the penalty comparable in
    magnitude to the log-likelihood, strongly discouraging rapid-switching
    degenerate solutions without requiring a hard minimum-duration threshold.

    Reference: Nystrup, P., Hansen, B.W. & Madsen, H. (2015–2019).
    "Regime-Based Asset Allocation", "Dynamic Allocation or Diversification",
    and related papers on penalised HMM estimation for financial time series.
    """
    if lam == 0.0:
        return model.score(X)
    penalty = sum(1.0 - model.transmat_[s, s] for s in range(model.n_components))
    return model.score(X) - lam * penalty


# ── Model training ────────────────────────────────────────────────────────────

def train_gaussian_hmm(
    X: np.ndarray,
    n_states: int = 2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    n_restarts: int = 5,
    duration_penalty: float = 0.0,
    vol_feature_idx: int = 1,
) -> hmm.GaussianHMM:
    """
    Train a GaussianHMM with multiple restarts and return the best model.

    Restart 0 uses volatility-quantile initialisation (_smart_init_hmm) to
    seed EM near an economically meaningful calm/volatile split.  Restarts
    1…n_restarts-1 use hmmlearn's default random k-means initialisation.

    Model selection uses penalised log-likelihood (see _penalised_score).

    Parameters
    ----------
    X                : (T, n_features) array — the observation sequence
    n_states         : number of hidden states
    n_restarts       : total restarts (restart 0 = smart init, rest = random)
    duration_penalty : λ in penalised LL; recommended ≈ len(X); 0 = disabled
    vol_feature_idx  : column index in X used as volatility proxy for smart init
                       (default 1 = abs_return in HMM_FEATURES)
    """
    best_model = None
    best_score = -np.inf

    for restart_idx in range(n_restarts):
        seed = random_state + restart_idx
        use_smart = (restart_idx == 0)
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
            init_params="" if use_smart else "stmc",
        )
        try:
            if use_smart:
                _smart_init_hmm(model, X, vol_feature_idx=vol_feature_idx, rng_seed=seed)
            model.fit(X)
            score = _penalised_score(model, X, duration_penalty)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as exc:
            logger.debug(
                "GaussianHMM restart %d (smart=%s) failed: %s", restart_idx, use_smart, exc
            )

    if best_model is None:
        raise RuntimeError("All GaussianHMM restarts failed.")

    logger.info(
        "GaussianHMM(n_states=%d) best penalised score: %.4f",
        n_states, best_score,
    )
    return best_model


def train_gmmhmm(
    X: np.ndarray,
    n_states: int = 2,
    n_mix: int = 2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    n_restarts: int = 5,
    duration_penalty: float = 0.0,
    vol_feature_idx: int = 1,
) -> hmm.GMMHMM:
    """
    Train a GMMHMM (Gaussian Mixture Model HMM) with multiple restarts.

    Each hidden state's emission is a mixture of n_mix Gaussians, giving
    the model more flexibility to capture heavy-tailed / multimodal
    within-state distributions.

    Restart 0 uses volatility-quantile initialisation (_smart_init_hmm) to
    seed EM near an economically meaningful calm/volatile split.  Restarts
    1…n_restarts-1 use hmmlearn's default random k-means initialisation.

    Model selection uses penalised log-likelihood (see _penalised_score).

    Parameters
    ----------
    duration_penalty : λ in penalised LL; recommended ≈ len(X); 0 = disabled
    vol_feature_idx  : column index in X used as volatility proxy for smart init
                       (default 1 = abs_return in HMM_FEATURES)
    """
    best_model = None
    best_score = -np.inf

    for restart_idx in range(n_restarts):
        seed = random_state + restart_idx
        use_smart = (restart_idx == 0)
        model = hmm.GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
            init_params="" if use_smart else "stmcw",
        )
        try:
            if use_smart:
                _smart_init_hmm(model, X, vol_feature_idx=vol_feature_idx, rng_seed=seed)
            model.fit(X)
            score = _penalised_score(model, X, duration_penalty)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as exc:
            logger.debug(
                "GMMHMM(n_mix=%d) restart %d (smart=%s) failed: %s",
                n_mix, restart_idx, use_smart, exc,
            )

    if best_model is None:
        raise RuntimeError(f"All GMMHMM(n_mix={n_mix}) restarts failed.")

    logger.info(
        "GMMHMM(n_states=%d, n_mix=%d) best penalised score: %.4f",
        n_states, n_mix, best_score,
    )
    return best_model


# ── Information criteria ──────────────────────────────────────────────────────

@dataclass
class ModelComparison:
    label: str
    model: object
    n_params: int
    train_log_likelihood: float
    val_log_likelihood: float
    n_obs_train: int
    n_obs_val: int
    bic_train: float = field(init=False)
    aic_train: float = field(init=False)
    bic_val: float = field(init=False)
    aic_val: float = field(init=False)

    def __post_init__(self) -> None:
        k = self.n_params
        self.bic_train = -2 * self.train_log_likelihood + k * np.log(self.n_obs_train)
        self.aic_train = -2 * self.train_log_likelihood + 2 * k
        self.bic_val = -2 * self.val_log_likelihood + k * np.log(self.n_obs_val)
        self.aic_val = -2 * self.val_log_likelihood + 2 * k


def compare_models(
    train_X: np.ndarray,
    val_X: np.ndarray,
    n_states: int = 2,
    n_mix_values: list[int] = [1, 2, 3],
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    n_restarts: int = 5,
    duration_penalty: float = 0.0,
) -> tuple[pd.DataFrame, list[ModelComparison]]:
    """
    Train and compare GaussianHMM and GMMHMM variants.

    n_mix = 1 is equivalent to GaussianHMM (single Gaussian per state).
    n_mix > 1 uses GMMHMM.

    Log-likelihood is the marginal P(O | model) computed via the forward
    algorithm — no labels required. BIC and AIC penalise model complexity.

    Parameters
    ----------
    duration_penalty : λ passed to train_gaussian_hmm / train_gmmhmm for
                       penalised restart selection (see _penalised_score).
                       Recommended: len(train_X). 0 = disabled.

    Returns
    -------
    summary_df   : DataFrame of results sorted by val BIC (ascending = better)
    comparisons  : list of ModelComparison objects (includes trained models)
    """
    n_features = train_X.shape[1]
    comparisons = []

    for n_mix in n_mix_values:
        if n_mix == 1:
            label = f"GaussianHMM(states={n_states})"
            model = train_gaussian_hmm(
                train_X, n_states=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter, random_state=random_state,
                n_restarts=n_restarts,
                duration_penalty=duration_penalty,
            )
            k = _count_gaussian_hmm_params(n_states, n_features, covariance_type)
        else:
            label = f"GMMHMM(states={n_states}, mix={n_mix})"
            model = train_gmmhmm(
                train_X, n_states=n_states, n_mix=n_mix,
                covariance_type=covariance_type,
                n_iter=n_iter, random_state=random_state,
                n_restarts=n_restarts,
                duration_penalty=duration_penalty,
            )
            k = _count_gmmhmm_params(n_states, n_mix, n_features, covariance_type)

        train_ll = model.score(train_X)
        val_ll = model.score(val_X)

        comp = ModelComparison(
            label=label,
            model=model,
            n_params=k,
            train_log_likelihood=train_ll,
            val_log_likelihood=val_ll,
            n_obs_train=len(train_X),
            n_obs_val=len(val_X),
        )
        comparisons.append(comp)
        logger.info(
            "%s — val LL=%.2f, val BIC=%.2f, val AIC=%.2f, k=%d",
            label, val_ll, comp.bic_val, comp.aic_val, k,
        )

    summary = pd.DataFrame([{
        "model": c.label,
        "n_params (k)": c.n_params,
        "train_LL": round(c.train_log_likelihood, 2),
        "val_LL": round(c.val_log_likelihood, 2),
        "train_BIC": round(c.bic_train, 2),
        "val_BIC": round(c.bic_val, 2),
        "train_AIC": round(c.aic_train, 2),
        "val_AIC": round(c.aic_val, 2),
    } for c in comparisons]).sort_values("val_BIC")

    best = summary.iloc[0]["model"]
    logger.info("Best model by val BIC: %s", best)

    return summary, comparisons


# ── Viterbi decoding ──────────────────────────────────────────────────────────

def viterbi_decode(
    model: hmm.GaussianHMM | hmm.GMMHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run Viterbi algorithm to get the most likely hidden state sequence.

    Returns an integer array of shape (T,) with values in {0, 1, ..., n_states-1}.
    """
    _, states = model.decode(X, algorithm="viterbi")
    return states


def forward_backward_proba(
    model: hmm.GaussianHMM | hmm.GMMHMM,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run the forward-backward algorithm to get soft state probabilities.

    Returns array of shape (T, n_states) where each row sums to 1.
    Used at inference time to produce the weighted ensemble prediction:
        y_final = P(calm | O) * y_calm + P(volatile | O) * y_volatile
    """
    return model.predict_proba(X)


# ── State relabeling ──────────────────────────────────────────────────────────

def identify_volatile_state(
    model: hmm.GaussianHMM | hmm.GMMHMM,
    X: np.ndarray,
    states: np.ndarray,
    feature_idx: int = 1,  # default: abs_return column index in HMM features
) -> dict[str, int]:
    """
    Determine which integer state label corresponds to the volatile regime
    by comparing the mean of *feature_idx* (abs_return) across states.

    Returns a dict: {'volatile': int, 'calm': int}
    """
    n_states = model.n_components
    state_means = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            state_means[s] = 0.0
        else:
            state_means[s] = X[mask, feature_idx].mean()

    volatile_state = max(state_means, key=state_means.get)
    calm_state = 1 - volatile_state  # assumes 2 states

    logger.info(
        "State identification — volatile=%d (mean abs_return=%.5f), calm=%d (mean abs_return=%.5f)",
        volatile_state, state_means[volatile_state],
        calm_state, state_means[calm_state],
    )
    return {"volatile": volatile_state, "calm": calm_state}


# ── Second-order Markov approximation ────────────────────────────────────────

def build_second_order_features(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Approximate a 2nd-order Markov HMM by augmenting the observation vector
    with its own lagged version, effectively giving the model access to
    (o_{t-1}, o_t) at each step.

    This is the feature-space approach to higher-order dependencies —
    simpler than full state-space augmentation and avoids the quadratic
    blowup in n_components while still capturing some temporal memory.

    Parameters
    ----------
    X   : (T, F) observation matrix
    lag : number of lags to append (default 1 → second order)

    Returns
    -------
    (T - lag, F * (lag + 1)) array with NaN rows dropped from the front.
    """
    lagged = np.hstack([X[lag:], X[:-lag]])
    return lagged


# ── Scaler ────────────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data.

    StandardScaler (zero-mean, unit-variance) is used for HMM inputs.
    RobustScaler was evaluated but produced larger scaled values for crisis
    periods (IQR < std), making the GMMHMM full-covariance EM unstable and
    causing convergence to degenerate solutions even with many restarts.
    StandardScaler keeps all features in a comparable range and yields
    stable EM convergence in practice.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model: object, path: str) -> None:
    joblib.dump(model, path)
    logger.info("Saved model → %s", path)


def load_model(path: str) -> object:
    model = joblib.load(path)
    logger.info("Loaded model ← %s", path)
    return model
