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


def _load(name: str, quick: bool) -> dict | None:
    fname = name.replace(".json", "_quick.json") if quick else name
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
    """Find crossover k where LEMON beats Bonneel; and the n where OR-Tools beats LEMON.

    Falls back to DEFAULT_THRESHOLDS when data is too sparse to decide.
    """
    bonneel_lemon = DEFAULT_THRESHOLDS["bonneel_lemon"]
    lemon_ortools = DEFAULT_THRESHOLDS["lemon_ortools"]

    ks_set: set[int] = set()
    for n_str in eff:
        ks_set.update(int(k) for k in eff[n_str])
    ks_sorted = sorted(ks_set)
    for k in ks_sorted:
        votes_lemon_wins = 0
        votes_total = 0
        for n_str, by_k in eff.items():
            cell = by_k.get(str(k), {})
            l = cell.get("lemon"); b = cell.get("bonneel")
            if (isinstance(l, dict) and isinstance(b, dict)
                and "error" not in l and "error" not in b
                and l.get("wall_time_s") is not None
                and b.get("wall_time_s") is not None):
                votes_total += 1
                if l["wall_time_s"] < b["wall_time_s"]:
                    votes_lemon_wins += 1
        if votes_total >= 3 and votes_lemon_wins / votes_total >= 0.7:
            bonneel_lemon = k
            break

    small_k_candidates = [k for k in ks_sorted if k <= 32]
    ns_sorted = sorted(int(n) for n in eff.keys())
    for n in ns_sorted:
        wins = 0; total = 0
        for k in small_k_candidates:
            cell = eff.get(str(n), {}).get(str(k), {})
            l = cell.get("lemon"); o = cell.get("ortools")
            if (isinstance(l, dict) and isinstance(o, dict)
                and "error" not in l and "error" not in o
                and l.get("wall_time_s") is not None
                and o.get("wall_time_s") is not None):
                total += 1
                if o["wall_time_s"] < l["wall_time_s"]:
                    wins += 1
        if total >= 3 and wins / total >= 0.7:
            lemon_ortools = n
            break

    return {"bonneel_lemon": int(bonneel_lemon),
            "lemon_ortools": int(lemon_ortools)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use *_quick.json files instead of full results")
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eff = _load("efficiency.json", args.quick)
    acc = _load("accuracy.json",   args.quick)
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
        print("skipping threshold derivation (--quick): use full sweep to "
              "update routing_thresholds.json", file=sys.stderr)
    else:
        thresholds = derive_thresholds(eff)
        out = RESULTS_DIR / "routing_thresholds.json"
        out.write_text(json.dumps(thresholds, indent=2) + "\n")
        print(f"wrote {out}: {thresholds}")
    print(f"figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
