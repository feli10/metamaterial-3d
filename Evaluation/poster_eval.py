"""
poster_eval.py — one experiment producing the poster's two core result figures.

For each sampled target it runs the full inverse-design loop and records target, achieved, confidence (fa), eps_e, and the target's distance to the training set. That single run yields both:
    Fig 1  achieved vs. target property accuracy
    Fig 2  design error vs. distance-to-training (and fa vs. error)

Targets are drawn by perturbing real training-property vectors by a controlled relative amount, so the sweep spans in-distribution to clearly out-of-distribution.

Writes a .npz of raw results next to the figures so plots can be redrawn without recomputing.

Usage: python Evaluation/poster_eval.py --n 60 --out Evaluation/poster_results
"""

import argparse
import os.path as osp
import sys
import time

import numpy as np

_repo = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(_repo, "web", "backend"))

import model_service as ms

EXPOSED = [0, 1, 7, 12, 16, 19, 21]
LABELS = ["vf", "C11", "C22", "C33", "C44", "C55", "C66"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of targets to evaluate")
    ap.add_argument("--model", default="s15v1")
    ap.add_argument("--out", default=osp.join(_repo, "Evaluation", "poster_results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    svc = ms.get_service(args.model)
    train = svc.all_props  # (n_cells, 22) real training property vectors

    # Perturbation scale per sample
    scales = np.linspace(0.0, 0.6, args.n)

    rows = []
    t_start = time.time()
    for i, scale in enumerate(scales):
        base = train[rng.integers(len(train))].copy()
        # Perturb only the exposed channels
        target_full = base.copy()
        pert = 1.0 + scale * rng.normal(size=len(EXPOSED))
        target_full[EXPOSED] = np.maximum(base[EXPOSED] * pert, 1e-4)

        # distance over the exposed channels
        d = np.sqrt(np.sum(
            (train[:, EXPOSED] - target_full[EXPOSED]) ** 2
            / (target_full[EXPOSED] + 1e-3) ** 2, axis=1))
        dist = float(d.min())

        target = [None] * 22
        for ch in EXPOSED:
            target[ch] = float(target_full[ch])

        g = svc.generate(target)
        e = ms.evaluate(g["voxels"])
        achieved = np.array(e["achieved"])

        rows.append({
            "requested": target_full[EXPOSED].copy(),
            "filled": np.array(g["filledTarget"])[EXPOSED],
            "achieved": achieved[EXPOSED],
            "fa": g["fa"], "eps_e": g["epsE"], "dist": dist, "scale": scale,
        })
        el = time.time() - t_start
        print(f"[{i+1}/{args.n}] scale={scale:.2f} dist={dist:.3f} fa={g['fa']:.4f} "f"| {el:.0f}s elapsed, ~{el/(i+1)*(args.n-i-1):.0f}s left", flush=True)

    requested = np.array([r["requested"] for r in rows])
    achieved = np.array([r["achieved"] for r in rows])
    fa = np.array([r["fa"] for r in rows])
    dist = np.array([r["dist"] for r in rows])
    eps_e = np.array([r["eps_e"] for r in rows])

    # Design error = mean relative error over the requested channels
    rel_err = np.abs(achieved - requested) / (np.abs(requested) + 1e-6)
    design_err = rel_err.mean(axis=1)

    np.savez(args.out + ".npz", requested=requested, achieved=achieved, fa=fa,
             dist=dist, eps_e=eps_e, design_err=design_err, labels=np.array(LABELS))
    print(f"\nsaved raw results -> {args.out}.npz")
    print(f"median design error: {np.median(design_err):.3f}")
    print(f"corr(dist, design_err) = {np.corrcoef(dist, design_err)[0,1]:.3f}")
    print(f"corr(fa, design_err)   = {np.corrcoef(fa, design_err)[0,1]:.3f}")

if __name__ == "__main__":
    main()
