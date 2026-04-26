"""Generate F0 (main-body headline figure): forecast trajectories + regime gate.

Panel (a): Realised vol target vs naive baseline, HAR-RV, variant A ensemble,
variant B ensemble over the 1,440-day common test window.
Panel (b): HMM gate posterior p_volatile for variants A and B over the same
window, with a 0.5 threshold.

Output: supplementary/figures/F0_main_panel.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "supplementary" / "figures" / "F0_main_panel.pdf"

COLOURS = {
    "target":  "#000000",
    "naive":   "#777777",
    "HAR-RV":  "#D35400",
    "A":       "#1E8449",
    "B":       "#6C3483",
}


def main() -> None:
    files = {
        "O":     DATA / "test_predictions_O.parquet",
        "A":     DATA / "test_predictions.parquet",
        "B":     DATA / "test_predictions_B.parquet",
        "H":     DATA / "test_predictions_H.parquet",
        "HAR-RV": DATA / "test_predictions_harrv.parquet",
    }
    common = None
    for f in files.values():
        df = pd.read_parquet(f)
        common = df.index if common is None else common.intersection(df.index)
    test_full = pd.read_parquet(DATA / "test.parquet")
    common = common.intersection(test_full.index)
    n = len(common)

    target = pd.read_parquet(files["A"])["target"].reindex(common)
    naive  = test_full["rolling_std_21"].reindex(common)
    harrv  = pd.read_parquet(files["HAR-RV"])["har_rv"].reindex(common)
    ensA   = pd.read_parquet(files["A"])["ensemble"].reindex(common)
    ensB   = pd.read_parquet(files["B"])["ensemble"].reindex(common)
    pvA    = pd.read_parquet(files["A"])["p_volatile"].reindex(common)
    pvB    = pd.read_parquet(files["B"])["p_volatile"].reindex(common)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 4.6),
                                         gridspec_kw={"height_ratios": [2.2, 1.0]},
                                         sharex=True)

    # Panel (a): forecasts
    ax_top.plot(common, target,  color=COLOURS["target"], linewidth=1.4, label="target (realized vol)")
    ax_top.plot(common, naive,   color=COLOURS["naive"],  linewidth=0.9, linestyle="--", label="naive (rolling 21d std)")
    ax_top.plot(common, harrv,   color=COLOURS["HAR-RV"], linewidth=0.9, label="HAR-RV")
    ax_top.plot(common, ensA,    color=COLOURS["A"],      linewidth=0.9, alpha=0.9, label="variant A ensemble")
    ax_top.plot(common, ensB,    color=COLOURS["B"],      linewidth=0.9, alpha=0.9, label="variant B ensemble")
    ax_top.set_ylabel("21-day forward realized vol")
    ax_top.set_title("(a) Forecasts vs realized vol on the 1,440-day common test window")
    ax_top.legend(loc="upper left", fontsize=7, ncol=3, frameon=False, columnspacing=1.0)
    ax_top.grid(linestyle=":", alpha=0.4)

    # Panel (b): regime gate
    ax_bot.plot(common, pvA, color=COLOURS["A"], linewidth=0.8, label=r"$p_{\mathrm{volatile}}$ (variant A)")
    ax_bot.plot(common, pvB, color=COLOURS["B"], linewidth=0.8, label=r"$p_{\mathrm{volatile}}$ (variant B)")
    ax_bot.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax_bot.set_ylim(-0.05, 1.08)
    ax_bot.set_ylabel(r"$p_{\mathrm{volatile}}$")
    ax_bot.set_title("(b) HMM gate posterior on the same window")
    ax_bot.legend(loc="upper left", fontsize=7, ncol=2, frameon=False, columnspacing=1.0)
    ax_bot.grid(linestyle=":", alpha=0.4)

    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}  (n={n})")


if __name__ == "__main__":
    main()
