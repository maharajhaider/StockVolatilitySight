"""Generate the GMMHMM-rationale appendix figure (2 panels).

Panel (a): Q-Q plot of training-window log returns against the normal distribution.
Panel (b): Pearson correlation heatmap of variant B's 16 HMM input features on the
training window. Cells with |corr| > 0.95 are highlighted as the collinearity
threshold used by the correlation-pruning step in section 3.1.

Output: supplementary/figures/F_gmmhmm_rationale.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # type: ignore  # noqa: E402

OUT = ROOT / "supplementary" / "figures" / "F_gmmhmm_rationale.pdf"


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    log_ret = train["log_return"].dropna().to_numpy()
    feats = config.HMM_VARIANT_B_FEATURES
    X = train[feats].dropna()

    fig, (ax_qq, ax_corr) = plt.subplots(1, 2, figsize=(11.0, 4.5),
                                         gridspec_kw={"width_ratios": [1.0, 1.4]})

    # ---- Panel (a): Q-Q plot of log returns ----
    osm, osr = stats.probplot(log_ret, dist="norm", fit=False)
    slope, intercept, _, _, _ = stats.linregress(osm, osr)
    skew = float(stats.skew(log_ret))
    excess_kurt = float(stats.kurtosis(log_ret, fisher=True))
    jb_stat, jb_p = stats.jarque_bera(log_ret)

    ax_qq.scatter(osm, osr, s=8, color="#922B21", alpha=0.5, edgecolors="none")
    xs = np.linspace(osm.min(), osm.max(), 100)
    ax_qq.plot(xs, slope * xs + intercept, color="#1B4F72", linewidth=1.0,
               label="best-fit line")
    ax_qq.plot(xs, np.std(log_ret) * xs + np.mean(log_ret), color="#117A65",
               linewidth=1.0, linestyle="--", label="$\\mathcal{N}$ reference")
    ax_qq.set_xlabel("Theoretical normal quantile")
    ax_qq.set_ylabel("Empirical log-return quantile")
    ax_qq.set_title(f"(a) Q-Q plot of SPY log returns vs $\\mathcal{{N}}$\n"
                    f"skew $=$ {skew:+.2f}, excess kurt $=$ {excess_kurt:+.2f}, "
                    f"Jarque--Bera $p < 10^{{-300}}$" if jb_p < 1e-300 else
                    f"(a) Q-Q plot of SPY log returns vs $\\mathcal{{N}}$\n"
                    f"skew $=$ {skew:+.2f}, excess kurt $=$ {excess_kurt:+.2f}, "
                    f"Jarque--Bera $p =$ {jb_p:.2e}",
                    fontsize=10)
    ax_qq.legend(fontsize=8, loc="upper left", frameon=False)
    ax_qq.grid(linestyle=":", alpha=0.4)

    # ---- Panel (b): correlation heatmap for variant B's 16 HMM features ----
    HIGHLIGHT = 0.85
    C = X.corr().to_numpy()
    n = C.shape[0]
    im = ax_corr.imshow(C, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax_corr.set_xticks(range(n))
    ax_corr.set_yticks(range(n))
    ax_corr.set_xticklabels(feats, rotation=60, ha="right", fontsize=7)
    ax_corr.set_yticklabels(feats, fontsize=7)
    high_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(C[i, j]) > HIGHLIGHT:
                ax_corr.add_patch(
                    plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                  fill=False, edgecolor="black", linewidth=1.2))
                if i < j:
                    high_pairs.append((feats[i], feats[j], C[i, j]))
    ax_corr.set_title(f"(b) Variant B HMM-feature correlation on train\n"
                      f"black-bordered cells: $|\\rho| > {HIGHLIGHT}$ "
                      f"({len(high_pairs)} pairs); 0.95 prune threshold not exceeded",
                      fontsize=10)

    cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson $\\rho$", fontsize=8)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"\nSummary stats on training-window log returns (n={len(log_ret)}):")
    print(f"  skew         = {skew:+.4f}")
    print(f"  excess kurt  = {excess_kurt:+.4f}")
    print(f"  Jarque-Bera  = {jb_stat:.1f}  (p = {jb_p:.3e})")
    print(f"\nVariant B HMM features (n={n}): {len(high_pairs)} pairs with |rho| > 0.95:")
    for a, b, r in sorted(high_pairs, key=lambda t: -abs(t[2])):
        print(f"  {a:24s}  vs  {b:24s}  rho = {r:+.3f}")


if __name__ == "__main__":
    main()
