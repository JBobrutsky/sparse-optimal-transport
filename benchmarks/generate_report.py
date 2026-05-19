# benchmarks/generate_report.py
"""Render benchmark figures from Bonneel-only sweep results.

Usage:
    python benchmarks/generate_report.py
    python benchmarks/generate_report.py --quick   # use *_quick.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def _load(name: str, quick: bool, mid: bool = False) -> dict | None:
    if quick:
        fname = name.replace(".json", "_quick.json")
    elif mid:
        fname = name.replace(".json", "_mid.json")
    else:
        fname = name
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _grid(data: dict, solver: str, field: str) -> tuple[list[int], list[int], np.ndarray]:
    ns = sorted(int(n) for n in data.keys())
    ks_set: set[int] = set()
    for n_str in data:
        ks_set.update(int(k) for k in data[n_str])
    ks = sorted(ks_set)
    grid = np.full((len(ns), len(ks)), np.nan, dtype=np.float64)
    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            cell = data.get(str(n), {}).get(str(k), {}).get(solver)
            if cell is None or not isinstance(cell, dict) or "error" in cell:
                continue
            v = cell.get(field)
            if v is None:
                continue
            grid[i, j] = v
    return ns, ks, grid


def _heatmap(ns, ks, ratio, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ratio = np.log10(ratio)
    im = ax.imshow(log_ratio, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_yticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("k (neighbors per source)")
    ax.set_ylabel("n (problem size)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(wall time ratio)")
    if np.any(np.isfinite(log_ratio)):
        try:
            ax.contour(log_ratio, levels=[0.0], colors="black", linewidths=2)
        except Exception:
            pass  # too few finite cells for a contour
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def render_efficiency_heatmaps(eff: dict) -> None:
    ns, ks, t_dense = _grid(eff, "bonneel_dense", "wall_time_s")
    _, _, t_sparse = _grid(eff, "bonneel_sparse", "wall_time_s")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_ds = t_dense / t_sparse

    _heatmap(ns, ks, ratio_ds,
             "bonneel_dense / bonneel_sparse wall-time ratio (contour = crossover)",
             FIGURES_DIR / "heatmap_dense_vs_sparse")


def render_line_plots(eff: dict) -> None:
    fixed_ks = [2, 8, 32, 128, 512]
    configs_markers = [
        ("bonneel_dense", "o"),
        ("bonneel_sparse", "s"),
        ("pot_reference", "x"),
    ]
    for k_target in fixed_ks:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for cfg, marker in configs_markers:
            xs, ys = [], []
            for n_str, by_k in eff.items():
                if str(k_target) not in by_k:
                    continue
                cell = by_k[str(k_target)].get(cfg)
                if cell is None or not isinstance(cell, dict) or "error" in cell:
                    continue
                if cell.get("wall_time_s") is None:
                    continue
                xs.append(int(n_str))
                ys.append(cell["wall_time_s"])
            if not xs:
                continue
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            ax.plot(xs, ys, marker=marker, label=cfg)
            plotted = True
        if not plotted:
            plt.close(fig); continue
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n"); ax.set_ylabel("wall time (s)")
        ax.set_title(f"Wall time vs n at k={k_target}")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        path = FIGURES_DIR / f"lineplot_k{k_target}"
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)


def render_accuracy_plot(acc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    any_plotted = False
    configs_markers = [
        ("bonneel_dense", "o"),
        ("bonneel_sparse", "s"),
        ("pot_reference", "x"),
    ]
    for cfg, marker in configs_markers:
        ks, errs, feas = [], [], []
        for n_str, by_k in acc.items():
            for k_str, by_cfg in by_k.items():
                cell = by_cfg.get(cfg)
                if cell is None or not isinstance(cell, dict) or "error" in cell:
                    continue
                if cell.get("rel_cost_err") is not None:
                    ks.append(int(k_str))
                    errs.append(cell["rel_cost_err"])
                feas.append(max(cell.get("feasibility_a", 0.0),
                                cell.get("feasibility_b", 0.0)))
        if ks:
            axes[0].scatter(ks, errs, marker=marker, label=cfg, alpha=0.6)
            any_plotted = True
        if feas:
            axes[1].scatter(range(len(feas)), feas, marker=marker, label=cfg, alpha=0.6)
            any_plotted = True
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("relative cost error vs reference")
    axes[0].set_title("Accuracy: relative cost error")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sample index"); axes[1].set_ylabel("||marginal residual||inf")
    axes[1].set_title("Feasibility: marginal residual")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if any_plotted:
        fig.savefig(FIGURES_DIR / "accuracy.pdf")
        fig.savefig(FIGURES_DIR / "accuracy.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use *_quick.json files instead of full results")
    ap.add_argument("--mid", action="store_true",
                    help="Use *_mid.json files (mid sweep)")
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eff = _load("efficiency.json", args.quick, args.mid)
    acc = _load("accuracy.json",   args.quick, args.mid)
    if eff is None:
        print("error: no efficiency JSON found; run bench_solvers.py first",
              file=sys.stderr)
        sys.exit(2)

    render_efficiency_heatmaps(eff)
    render_line_plots(eff)
    if acc is not None:
        render_accuracy_plot(acc)
    else:
        print("warning: no accuracy JSON found; skipping accuracy plot",
              file=sys.stderr)

    print(f"figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
