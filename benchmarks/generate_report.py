# benchmarks/generate_report.py
"""Render benchmark figures and derive routing thresholds.

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

DEFAULT_THRESHOLDS = {"bonneel_lemon": 128, "lemon_ortools": 1_000_000}


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


def _load_best_efficiency() -> tuple[dict | None, str]:
    """Load the best available efficiency JSON (full > mid > quick)."""
    for suffix, label in [("", "full"), ("_mid", "mid"), ("_quick", "quick")]:
        p = RESULTS_DIR / f"efficiency{suffix}.json"
        if p.exists():
            return json.loads(p.read_text()), label
    return None, "none"


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
    _, _, t_bon = _grid(eff, "bonneel", "wall_time_s")
    ns, ks, t_lem = _grid(eff, "lemon", "wall_time_s")
    _, _, t_ort = _grid(eff, "ortools", "wall_time_s")

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_lb = t_lem / t_bon
        ratio_ol = t_ort / t_lem

    _heatmap(ns, ks, ratio_lb,
             "LEMON / Bonneel wall-time ratio (contour = crossover)",
             FIGURES_DIR / "heatmap_lemon_vs_bonneel")
    _heatmap(ns, ks, ratio_ol,
             "OR-Tools / LEMON wall-time ratio (contour = crossover)",
             FIGURES_DIR / "heatmap_ortools_vs_lemon")


def render_line_plots(eff: dict) -> None:
    fixed_ks = [2, 8, 32, 128, 512]
    for k_target in fixed_ks:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for solver, marker in [("bonneel", "o"), ("lemon", "s"),
                               ("ortools", "^"), ("pot_reference", "x")]:
            xs, ys = [], []
            for n_str, by_k in eff.items():
                if str(k_target) not in by_k:
                    continue
                cell = by_k[str(k_target)].get(solver)
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
            ax.plot(xs, ys, marker=marker, label=solver)
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
    for solver, marker in [("lemon", "s"), ("ortools", "^")]:
        ks, errs, feas = [], [], []
        for n_str, by_k in acc.items():
            for k_str, by_solver in by_k.items():
                cell = by_solver.get(solver)
                if cell is None or not isinstance(cell, dict) or "error" in cell:
                    continue
                if cell.get("rel_cost_err") is not None:
                    ks.append(int(k_str))
                    errs.append(cell["rel_cost_err"])
                feas.append(max(cell.get("feasibility_a", 0.0),
                                cell.get("feasibility_b", 0.0)))
        if ks:
            axes[0].scatter(ks, errs, marker=marker, label=solver, alpha=0.6)
            any_plotted = True
        if feas:
            axes[1].scatter(range(len(feas)), feas, marker=marker, label=solver, alpha=0.6)
            any_plotted = True
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("relative cost error vs POT")
    axes[0].set_title("Accuracy: relative cost error")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sample index"); axes[1].set_ylabel("‖marginal residual‖∞")
    axes[1].set_title("Feasibility: marginal residual")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if any_plotted:
        fig.savefig(FIGURES_DIR / "accuracy.pdf")
        fig.savefig(FIGURES_DIR / "accuracy.png", dpi=150)
    plt.close(fig)


def derive_thresholds(eff: dict) -> dict:
    """Derive routing thresholds from an efficiency sweep.

    Returns
    -------
    dict with two keys:
      bonneel_lemon : smallest k at which bonneel beats lemon (any n where
        both ran); falls back to 128 if no crossover is observed.
      lemon_ortools : smallest n at which ortools beats lemon for at least
        one k (both must have a non-null wall_time_s); falls back to 1_000_000.
    """
    bonneel_lemon = None
    for n_str in sorted(eff, key=int):
        by_k = eff[n_str]
        for k_str, by_s in sorted(by_k.items(), key=lambda kv: int(kv[0])):
            wb = (by_s.get('bonneel') or {}).get('wall_time_s')
            wl = (by_s.get('lemon') or {}).get('wall_time_s')
            if wb is not None and wl is not None and wb < wl:
                k = int(k_str)
                bonneel_lemon = k if bonneel_lemon is None else min(bonneel_lemon, k)
                break

    lemon_ortools = None
    for n_str in sorted(eff, key=int):
        by_k = eff[n_str]
        crossed = False
        for k_str, by_s in by_k.items():
            wl = (by_s.get('lemon') or {}).get('wall_time_s')
            wo = (by_s.get('ortools') or {}).get('wall_time_s')
            if wl is not None and wo is not None and wo < wl:
                crossed = True
                break
        if crossed:
            n = int(n_str)
            lemon_ortools = n if lemon_ortools is None else min(lemon_ortools, n)

    return {
        'bonneel_lemon': bonneel_lemon if bonneel_lemon is not None else 128,
        'lemon_ortools': lemon_ortools if lemon_ortools is not None else 1_000_000,
    }


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

    if args.quick:
        print("skipping threshold derivation (--quick): use --mid or full sweep to "
              "update routing_thresholds.json", file=sys.stderr)
    else:
        # For threshold derivation, use the best available data (full > mid > quick).
        eff_for_thresh, label = _load_best_efficiency()
        if eff_for_thresh is None:
            eff_for_thresh = eff
            label = "current"
        print(f"deriving thresholds from {label} efficiency data", file=sys.stderr)
        thresholds = derive_thresholds(eff_for_thresh)
        out = RESULTS_DIR / "routing_thresholds.json"
        out.write_text(json.dumps(thresholds, indent=2) + "\n")
        print(f"wrote {out}: {thresholds}")
    print(f"figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
