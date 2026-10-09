#!/usr/bin/env python3
"""Paper-build helper (not a numbered pipeline stage): draws the graphical abstract.

Left: the pipeline and how it is evaluated. Right: the held-out C-index of k-means
on the MOFA factors at each fixed k, averaged over the ten repeated splits of stage 22,
with the value silhouette selection chose (k = 2 on every split) and the range the
survival rule chose (k = 4 to 8).

The journal asks for 531 x 1328 pixels (h x w) or proportionally more, readable at
5 x 13 cm, so the figure is drawn at that physical size and saved at three times the
pixel count.

Outputs: outputs/figures/graphical_abstract.{pdf,png,tif}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#cde2fb"
BOX_FILL, BOX_EDGE = "#f0efec", "#b9b8b2"

CM = 1 / 2.54
W, H = 13.0 * CM, 13.0 * 531 / 1328 * CM


def main():
    summ = pd.read_csv(TABLES / "fixed_k_summary.csv")
    fixed = summ[summ.quantity.str.startswith("fixed k = ")].copy()
    fixed["k"] = fixed.quantity.str.replace("fixed k = ", "").astype(int)
    fixed = fixed[fixed.k <= 8].sort_values("k")   # the search range, as in Table 4
    surv = summ.loc[summ.quantity == "survival-guided k, searched 2-8", "mean_c_index"].item()
    sil = fixed.loc[fixed.k == 2, "mean_c_index"].item()
    per = pd.read_csv(TABLES / "repeated_splits_per_split.csv")
    better = int((per.c_survival > per.c_silhouette).sum())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    fig = plt.figure(figsize=(W, H), facecolor="white")

    # ---- left: pipeline ---------------------------------------------------------------
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(3, 92, "A pipeline for multi-omic subtyping", fontsize=6.6, weight="bold", color=INK, va="center")
    steps = ["mRNA, copy\nnumber and\nmutations", "Pathway, arm\nand mutation\nfeatures",
             "MOFA latent\nfactors", "k chosen by\nsurvival, then\nk-means"]
    bw, gap, x0, y0, bh = 21.5, 3.0, 3.0, 50.0, 30.0
    for i, s in enumerate(steps):
        x = x0 + i * (bw + gap)
        hi = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=2.2",
                                    fc="#e3eefc" if hi else BOX_FILL, ec=BLUE if hi else BOX_EDGE, lw=0.6))
        ax.text(x + bw / 2, y0 + bh / 2, s, ha="center", va="center", fontsize=5.4, color=INK, linespacing=1.15)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + bw + gap - 0.3, y0 + bh / 2), xytext=(x + bw + 0.3, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=5))
    ax.add_patch(FancyBboxPatch((x0, 9), 4 * bw + 3 * gap, 30, boxstyle="round,pad=0,rounding_size=2.2",
                                fc="white", ec=BOX_EDGE, lw=0.6))
    ax.text(x0 + 2.5, 32, "Evaluation", fontsize=6.0, weight="bold", color=INK, va="center")
    ax.text(x0 + 2.5, 18.5,
            "METABRIC (n = 1860)  ·  held-out C-index  ·  10 repeated splits\n"
            "simulation  ·  stability  ·  treatment effects  ·  TCGA-BRCA",
            fontsize=5.6, color=INK2, va="center", linespacing=1.35)

    # ---- right: the result --------------------------------------------------------------
    ax2 = fig.add_axes([0.665, 0.20, 0.315, 0.56])
    ax2.axvspan(3.6, 8.5, color=BAND, lw=0, zorder=0)
    ax2.plot(fixed.k, fixed.mean_c_index, color=BLUE, lw=1.0, zorder=2)
    ax2.scatter(fixed.k, fixed.mean_c_index, s=9, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax2.scatter([2], [sil], s=16, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=4)
    ax2.annotate("silhouette\nchose k = 2", xy=(2, sil), xytext=(2.35, 0.5415), fontsize=5.4, color=INK,
                 va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4))
    ax2.text(6.05, 0.5705, "survival rule\nchose k = 4 to 8", ha="center", va="center", fontsize=5.4, color=INK)
    ax2.set_xlim(1.5, 8.5); ax2.set_ylim(0.538, 0.576)
    ax2.set_xticks(range(2, 9)); ax2.set_yticks([0.54, 0.55, 0.56, 0.57])
    ax2.tick_params(colors=INK2, labelsize=5.4)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(MUTED)
    ax2.grid(axis="y", color="#e6e5e1", lw=0.4, zorder=0)
    ax2.set_xlabel("Number of clusters k", fontsize=5.6, color=INK2, labelpad=1.5)
    ax2.set_ylabel("Held-out C-index\n(mean of 10 splits)", fontsize=5.2, color=INK2, labelpad=1.5)
    fig.text(0.615, 0.92, "Choosing k by survival helps", fontsize=6.6, weight="bold", color=INK, va="center")
    fig.text(0.615, 0.825, f"Held-out C-index {surv:.3f} vs {sil:.3f} for silhouette,\nbetter in {better} of 10 repeated splits",
             fontsize=5.4, linespacing=1.25, color=INK2, va="center")

    FIGS.mkdir(parents=True, exist_ok=True)
    dpi = 3 * 1328 / W
    fig.savefig(FIGS / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(FIGS / "graphical_abstract.png", dpi=dpi, facecolor="white")
    fig.savefig(FIGS / "graphical_abstract.tif", dpi=dpi, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    print("wrote graphical_abstract.{pdf,png,tif}")


if __name__ == "__main__":
    main()
