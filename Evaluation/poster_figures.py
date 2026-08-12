"""
poster_figures.py — draw the poster figures from poster_eval.py's saved results.

Fig 1  fig1_property_accuracy.png : achieved vs. requested properties
Fig 2  fig2_error_vs_distance.png : design error vs. distance-to-training data

Usage: python Evaluation/poster_figures.py [--res Evaluation/poster_results.npz]
"""

import argparse
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN = "#1D9E75"
CRIMSON = "#A6192E"
SLATE = "#1B2430"
MUTED = "#6B7482"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#CAD2DC",
    "axes.labelcolor": SLATE,
    "text.color": SLATE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def fig1_property_accuracy(d, out):
    """
    Achieved vs. requested, one panel per property channel.

    Panels are laid out by physical meaning: the three AXIAL stiffnesses on the top row, the three SHEAR stiffnesses on the second, and volume fraction alone in the bottom-right corner (it is not a stiffness, so it reads as its own category).
    """
    requested, achieved, labels = d["requested"], d["achieved"], list(d["labels"])
    layout = ["C11", "C22", "C33", None, "C44", "C55", "C66", "vf"]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for slot, ax in zip(layout, axes.ravel()):
        if slot is None:
            ax.axis("off")
            continue
        k = labels.index(slot)
        x, y = requested[:, k], achieved[:, k]
        lo = 0.0
        hi = max(x.max(), y.max()) * 1.08
        ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1, zorder=1) # ideal
        ax.scatter(x, y, s=26, color=GREEN, alpha=0.75, edgecolor="white",
                   linewidth=0.4, zorder=2)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{labels[k]}   (r = {r:.3f})", fontsize=11, color=SLATE)
        ax.set_xlabel("requested")
        ax.set_ylabel("achieved (FEM)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    fig.suptitle("Achieved vs. requested properties - dashed line is perfect agreement",
                 fontsize=13, color=SLATE)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def fig1_compact(d, out):
    """
    Poster-legible 2x2 version of Fig 1: two axial channels over two shear channels.

    The full 7-panel figure shrinks to illegibility in a half-column at poster scale, so this shows the four channels that carry the axial-vs-shear finding, at larger type and marker size, and reports the remaining channels' correlations as a compact caption instead.
    """
    requested, achieved, labels = d["requested"], d["achieved"], list(d["labels"])
    shown = ["C11", "C33", "C44", "C66"] # row 1 axial, row 2 shear

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6))
    for slot, ax in zip(shown, axes.ravel()):
        k = labels.index(slot)
        x, y = requested[:, k], achieved[:, k]
        hi = max(x.max(), y.max()) * 1.08
        ax.plot([0, hi], [0, hi], "--", color=MUTED, lw=1.4, zorder=1)
        ax.scatter(x, y, s=60, color=GREEN, alpha=0.8, edgecolor="white",
                   linewidth=0.6, zorder=2)
        r = np.corrcoef(x, y)[0, 1]
        kind = "axial" if slot in ("C11", "C22", "C33") else "shear"
        ax.set_title(f"{slot}  ({kind})\nr = {r:.2f}", fontsize=15, color=SLATE)
        ax.set_xlabel("requested", fontsize=13)
        ax.set_ylabel("achieved (FEM)", fontsize=13)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=11)
        ax.locator_params(nbins=4) # fewer ticks = readable from a distance
        ax.grid(alpha=0.25)

    # Report the channels not plotted so nothing is hidden.
    rest = [f"{n} {np.corrcoef(requested[:, labels.index(n)], achieved[:, labels.index(n)])[0,1]:.2f}"
            for n in ("C22", "C55", "vf")]
    fig.text(0.5, 0.005, "also measured:  " + "   |   ".join(rest),
             ha="center", fontsize=11, color=MUTED)
    fig.suptitle("Achieved vs. requested properties\ndashed line = perfect agreement",
                 fontsize=16, color=SLATE)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out)
    print("wrote", out)


def fig2_error_vs_distance(d, out):
    """
    Design error vs. distance to training data - the reliability/trust figure.

    Error metric note: we use MEAN ABSOLUTE error over the requested channels, not relative error. Relative error divides by the requested value, so targets whose requested property is near zero produce astronomically large percentages — a handful of such points would dominate the plot and manufacture a spurious correlation. Absolute error is on a common,
    physically meaningful scale here (all properties are normalized to base material E=1). Spearman (rank) correlation is reported alongside Pearson because it is not distorted by the remaining outliers.
    """
    requested, achieved = d["requested"], d["achieved"]
    abs_err = np.abs(achieved - requested).mean(axis=1)
    dist, fa = d["dist"], d["fa"]
    # One sampled target sits exactly on a training point (dist == 0), which the log axis and the log-space fit can't represent. Floor it just below the smallest real distance.
    pos = dist[dist > 0]
    dist = np.maximum(dist, pos.min() * 0.5 if len(pos) else 1e-3)

    # Single colour, no fa colour axis: post-generation fa is saturated (generate() optimises it), so encoding it here was a near-constant scale that added a legend without adding information. The confidence-vs-distance relationship is shown properly in its own figure.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dist, abs_err, s=52, color=GREEN, alpha=0.8,
               edgecolor="white", linewidth=0.5)

    pear = np.corrcoef(dist, abs_err)[0, 1]
    r_d = np.argsort(np.argsort(dist))
    r_e = np.argsort(np.argsort(abs_err))
    spear = np.corrcoef(r_d, r_e)[0, 1]

    # Log x: distances span orders of magnitude (near-training to far OOD), so a linear axis crushes the in-distribution bulk into the left edge.
    ax.set_xscale("log")

    # No fitted trend line: the spread is heteroscedastic (tight near the training data, fanning out far from it), so a straight fit would overstate how well a line describes it. The rank correlation states the monotonic relationship without implying a shape.
    ax.annotate(f"Spearman ρ = {spear:.2f}", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=12, color=SLATE, fontweight="bold")
    ax.set_xlabel("distance of target from training data (log scale)")
    ax.set_ylabel("design error (mean absolute, normalized units)")
    ax.set_title("Design error grows as targets leave the training distribution", fontsize=12, color=SLATE)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)
    print(f"   fig2: Pearson {pear:.3f} | Spearman {spear:.3f} | "
          f"median abs err {np.median(abs_err):.4f}")

def main():
    ap = argparse.ArgumentParser()
    _repo = osp.dirname(osp.dirname(osp.abspath(__file__)))
    ap.add_argument("--res", default=osp.join(_repo, "Evaluation", "poster_results.npz"))
    args = ap.parse_args()
    d = np.load(args.res)
    base = args.res.replace(".npz", "")
    fig1_property_accuracy(d, base + "_fig1_property_accuracy.png")
    fig1_compact(d, base + "_fig1_compact.png")
    fig2_error_vs_distance(d, base + "_fig2_error_vs_distance.png")

    print(f"\nn = {len(d['fa'])} designs")
    print(f"median design error : {np.median(d['design_err'])*100:.1f}%")
    print(f"corr(dist, error)   : {np.corrcoef(d['dist'], d['design_err'])[0,1]:.3f}")
    print(f"fa range            : {d['fa'].min():.4f} – {d['fa'].max():.4f}")

if __name__ == "__main__":
    main()
