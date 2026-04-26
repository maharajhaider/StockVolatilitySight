"""Generate F4: HMM-posterior p_volatile across the full timeline for variants O, Os, A, B.

Each panel shows the full-timeline p_volatile trajectory with train/val/test boundaries
and a per-panel header summarising the test-window mean, share above 0.5, and lag-1
autocorrelation. The variant-O label correctly states the BIC-selected mix=3.

Output: supplementary/figures/F4_p_volatile_over_time.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "supplementary" / "figures" / "F4_p_volatile_over_time.pdf"

TRAIN_END = pd.Timestamp("2015-12-31")
VAL_END = pd.Timestamp("2019-12-31")

VARIANTS = [
    ("O",  "Variant O (GMMHMM mix=3, 11 features)",                 "data/processed/regime_probabilities_O.parquet",  "data/processed/test_predictions_O.parquet"),
    ("Os", "Variant Os (plain GaussianHMM, 11 features)",           "data/processed/regime_probabilities_Os.parquet", "data/processed/test_predictions_Os.parquet"),
    ("A",  "Variant A (GMMHMM mix=2, +AAII sentiment, 13 features)", "data/processed/regime_probabilities.parquet",   "data/processed/test_predictions.parquet"),
    ("B",  "Variant B (GMMHMM mix=2, +VIX-family, 16 features)",    "data/processed/regime_probabilities_B.parquet",  "data/processed/test_predictions_B.parquet"),
]

COLOURS = {"O": "#922B21", "Os": "#1B4F72", "A": "#1E8449", "B": "#6C3483"}


def main() -> None:
    fig, axes = plt.subplots(len(VARIANTS), 1, figsize=(11.0, 6.5), sharex=True)
    fig.suptitle("HMM-posterior $p_{\\mathrm{volatile}}$ across the full timeline, by HMM variant. "
                 "O vs Os: same features, different HMM emission model class.",
                 fontsize=9, y=0.995)

    for ax, (label, title_prefix, full_path, test_path) in zip(axes, VARIANTS):
        rp = pd.read_parquet(ROOT / full_path)
        td = pd.read_parquet(ROOT / test_path)
        pv_full = rp["p_volatile"]
        pv_test = td["p_volatile"].values
        ac_test = float(np.corrcoef(pv_test[:-1], pv_test[1:])[0, 1])
        mean_test = float(pv_test.mean())
        share_test = float((pv_test > 0.5).mean())
        header = (f"{title_prefix} test: mean={mean_test:.2f}, share $>$ 0.5={share_test*100:.0f}\\%, "
                  f"lag-1 autocorr={ac_test:.2f}")
        ax.set_title(header, fontsize=8, loc="left")
        ax.fill_between(pv_full.index, 0, pv_full.values, color=COLOURS[label], alpha=0.35, linewidth=0)
        ax.plot(pv_full.index, pv_full.values, color=COLOURS[label], linewidth=0.55)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel(r"$p_{\mathrm{vol}}$", fontsize=8)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(linestyle=":", alpha=0.3)
        # Train / val / test boundaries
        for boundary, name, x_offset in [(TRAIN_END, "TRAIN", -0.08),
                                         (VAL_END, "VAL", -0.04)]:
            ax.axvline(boundary, color="#444444", linestyle=":", linewidth=0.7)
        if label == "O":  # add labels only on top panel to avoid clutter
            for boundary, name in [(pv_full.index.min(), "TRAIN"),
                                   (TRAIN_END + pd.Timedelta(days=10), "VAL"),
                                   (VAL_END + pd.Timedelta(days=10), "TEST")]:
                ax.text(boundary, 1.05, name, fontsize=7, color="#444444", va="bottom")

    axes[-1].set_xlabel("Date (full timeline: train + val + test)", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
