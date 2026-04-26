"""Generate F5: side-by-side comparison of GMMHMM (O/A/B) vs GaussianHMM (Os/As/Bs).

Two panels:
  (a) Test-set MAPE per variant pair, with the variant's LSTM-only baseline as a
      dashed reference line.
  (b) Lag-1 autocorrelation of the test-window p_volatile posterior, the regime-
      quality proxy used in section 4.5.

Output: supplementary/figures/F5_emission_comparison.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "supplementary" / "figures" / "F5_emission_comparison.pdf"

PAIRS = [
    ("O",  "Os"),
    ("A",  "As"),
    ("B",  "Bs"),
]

FILES = {
    "O":  DATA / "test_predictions_O.parquet",
    "A":  DATA / "test_predictions.parquet",
    "B":  DATA / "test_predictions_B.parquet",
    "Os": DATA / "test_predictions_Os.parquet",
    "As": DATA / "test_predictions_As.parquet",
    "Bs": DATA / "test_predictions_Bs.parquet",
}

# Match colour palette already established in F1-F4.
COLOURS = {
    "O":  "#922B21",  # dark red
    "A":  "#1E8449",  # dark green
    "B":  "#6C3483",  # dark purple
    "Os": "#E6B0AA",  # light red
    "As": "#A9DFBF",  # light green
    "Bs": "#D2B4DE",  # light purple
}


def metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["target"].to_numpy()
    yhat = df["ensemble"].to_numpy()
    base = df["baseline"].to_numpy()
    mape = float(np.mean(np.abs((yhat - y) / y)) * 100.0)
    bmape = float(np.mean(np.abs((base - y) / y)) * 100.0)
    pv = df["p_volatile"].to_numpy()
    autocorr = float(np.corrcoef(pv[:-1], pv[1:])[0, 1])
    return {"mape": mape, "baseline_mape": bmape, "autocorr": autocorr}


def main() -> None:
    rows = {v: metrics(pd.read_parquet(p)) for v, p in FILES.items()}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 5.4))

    n_pairs = len(PAIRS)
    bar_w = 0.36
    centres = np.arange(n_pairs)

    # Panel (a): MAPE
    for offset, suffix_idx in enumerate([0, 1]):  # mixture, then gaussian
        labels = [pair[suffix_idx] for pair in PAIRS]
        vals = [rows[v]["mape"] for v in labels]
        x = centres + (offset - 0.5) * bar_w
        for xi, vi, lab in zip(x, vals, labels):
            ax_top.bar(xi, vi, width=bar_w, color=COLOURS[lab],
                       edgecolor="black", linewidth=0.6, label=lab)
            ax_top.text(xi, vi + 0.15, f"{vi:.2f}", ha="center", va="bottom",
                        fontsize=8)

    # Variant-specific baseline dashes (over each pair).
    for i, (mix, gauss) in enumerate(PAIRS):
        b = rows[mix]["baseline_mape"]  # same baseline for the pair (LSTM-only on the same feature set)
        ax_top.hlines(b, centres[i] - 0.5, centres[i] + 0.5,
                      colors="#444444", linestyles="--", linewidth=1.0)
        ax_top.text(centres[i] + 0.48, b, f" base {b:.2f}",
                    ha="left", va="center", fontsize=7, color="#444444")

    ax_top.set_xticks(centres)
    ax_top.set_xticklabels([f"{m} vs {g}" for m, g in PAIRS])
    ax_top.set_ylabel("Test MAPE (\\%)")
    ax_top.set_title("(a) Ensemble accuracy: GMMHMM (left of each pair) vs GaussianHMM (right)")
    ax_top.set_ylim(0, max(rows[v]["mape"] for v in rows) * 1.18)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)

    # Build a compact legend for the pairing convention.
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[v], ec="black", lw=0.6) for v in
               ["O", "Os", "A", "As", "B", "Bs"]]
    ax_top.legend(handles, ["O", "Os", "A", "As", "B", "Bs"],
                  ncol=6, loc="upper right", fontsize=7, frameon=False,
                  handlelength=1.2, columnspacing=0.8)

    # Panel (b): regime persistence
    for offset, suffix_idx in enumerate([0, 1]):
        labels = [pair[suffix_idx] for pair in PAIRS]
        vals = [rows[v]["autocorr"] for v in labels]
        x = centres + (offset - 0.5) * bar_w
        for xi, vi, lab in zip(x, vals, labels):
            ax_bot.bar(xi, vi, width=bar_w, color=COLOURS[lab],
                       edgecolor="black", linewidth=0.6)
            ax_bot.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom",
                        fontsize=8)

    ax_bot.set_xticks(centres)
    ax_bot.set_xticklabels([f"{m} vs {g}" for m, g in PAIRS])
    ax_bot.set_ylabel(r"Lag-1 autocorr of $p_{\mathrm{volatile}}$")
    ax_bot.set_title("(b) Regime persistence on the test window (higher $=$ more persistent regimes)")
    ax_bot.set_ylim(0, 1.1)
    ax_bot.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print("Summary:")
    for v in ["O", "Os", "A", "As", "B", "Bs"]:
        print(f"  {v:<3}  MAPE={rows[v]['mape']:6.3f}  base={rows[v]['baseline_mape']:6.3f}  autocorr={rows[v]['autocorr']:.3f}")


if __name__ == "__main__":
    main()
