"""Generate F_variant_O_hmm_diagnostic: 2-panel before/after for variant O HMM.

Top: variant O as currently used (BIC-optimal GMMHMM(mix=3)).
Bottom: same 11 features refit as plain GaussianHMM (variant Os).

Output: supplementary/figures/F_variant_O_hmm_diagnostic.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "supplementary" / "figures" / "F_variant_O_hmm_diagnostic.pdf"


def stats(test_pred_path: str) -> tuple[float, float, float]:
    pv = pd.read_parquet(ROOT / test_pred_path)["p_volatile"].values
    return float(pv.mean()), float((pv > 0.5).mean()), float(np.corrcoef(pv[:-1], pv[1:])[0, 1])


def main() -> None:
    rp_O  = pd.read_parquet(ROOT / "data/processed/regime_probabilities_O.parquet")
    rp_Os = pd.read_parquet(ROOT / "data/processed/regime_probabilities_Os.parquet")

    O_mean, O_share, O_ac = stats("data/processed/test_predictions_O.parquet")
    Os_mean, Os_share, Os_ac = stats("data/processed/test_predictions_Os.parquet")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 4.0), sharex=True)
    fig.suptitle("Variant O HMM diagnostic: GMMHMM(mix=3) converges to a non-regime basin "
                 f"(autocorr {O_ac:.1f}); GaussianHMM finds clean regimes (autocorr {Os_ac:.2f})",
                 fontsize=9, y=0.995)

    ax_top.set_title(
        f"BEFORE: variant O HMM as currently saved (GMMHMM mix=3, BIC winner)  "
        f"rapid switching [test mean={O_mean:.2f}, share$>$0.5={O_share*100:.0f}\\%, "
        f"lag-1 autocorr={O_ac:.2f}]",
        fontsize=8, loc="left",
    )
    ax_top.fill_between(rp_O.index, 0, rp_O["p_volatile"].values, color="#922B21", alpha=0.35, linewidth=0)
    ax_top.plot(rp_O.index, rp_O["p_volatile"].values, color="#922B21", linewidth=0.55)
    ax_top.set_ylim(-0.05, 1.08)
    ax_top.set_ylabel(r"$p_{\mathrm{vol}}$", fontsize=8)
    ax_top.set_yticks([0.0, 0.5, 1.0])
    ax_top.grid(linestyle=":", alpha=0.3)

    ax_bot.set_title(
        f"AFTER: variant O HMM refit as plain GaussianHMM   "
        f"clean regime persistence [test mean={Os_mean:.2f}, share$>$0.5={Os_share*100:.0f}\\%, "
        f"lag-1 autocorr={Os_ac:.2f}]",
        fontsize=8, loc="left",
    )
    ax_bot.fill_between(rp_Os.index, 0, rp_Os["p_volatile"].values, color="#1B4F72", alpha=0.35, linewidth=0)
    ax_bot.plot(rp_Os.index, rp_Os["p_volatile"].values, color="#1B4F72", linewidth=0.55)
    ax_bot.set_ylim(-0.05, 1.08)
    ax_bot.set_ylabel(r"$p_{\mathrm{vol}}$", fontsize=8)
    ax_bot.set_yticks([0.0, 0.5, 1.0])
    ax_bot.grid(linestyle=":", alpha=0.3)

    ax_bot.set_xlabel("Date (full timeline: train + val + test)", fontsize=8)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
