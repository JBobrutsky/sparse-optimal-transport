"""benchmarks/report.py — produce 4 PNG figures from bench_{tag}.json.

Usage:
    python benchmarks/report.py            # reads bench.json
    python benchmarks/report.py --quick    # reads bench_quick.json
    python benchmarks/report.py --mid      # reads bench_mid.json
    python benchmarks/report.py --print-tables  # also prints Markdown tables
"""

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

SOLVER_STYLE = {
    "sparse_ot": {"color": "tab:blue",   "marker": "o", "label": "sparse-ot"},
    "pot":       {"color": "tab:orange", "marker": "s", "label": "POT (ot.emd)"},
    "ortools":   {"color": "tab:green",  "marker": "^", "label": "OR-Tools"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(tag: str) -> dict:
    suffix = f"_{tag}" if tag != "full" else ""
    p = RESULTS_DIR / f"bench{suffix}.json"
    if not p.exists():
        print(f"error: {p} not found", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text())


def _fmt_time(t) -> str:
    if t is None:
        return "—"
    if t < 0.001:
        return f"{t*1000:.2f}ms"
    if t < 1:
        return f"{t:.3f}s"
    if t < 60:
        return f"{t:.1f}s"
    return f"~{t/60:.0f}min"


def _placeholder(out_path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.axis("off")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: dense_cold.png
# ---------------------------------------------------------------------------

def fig_dense_cold(cells: list, figures_dir: Path) -> None:
    out_path = figures_dir / "dense_cold.png"

    # Group by solver -> {n: (wall_s, extrapolated)}
    solver_data: dict[str, dict] = {}
    for c in cells:
        if c["scenario"] != "dense_cold":
            continue
        solver = c["solver"]
        if solver not in SOLVER_STYLE:
            continue
        if c["wall_s"] is None:
            continue
        solver_data.setdefault(solver, {})[c["n"]] = (c["wall_s"], c["extrapolated"])

    fig, ax = plt.subplots(figsize=(7, 5))

    for solver, style in SOLVER_STYLE.items():
        if solver not in solver_data:
            continue
        pts = solver_data[solver]
        solid_ns = sorted(n for n, (_, ext) in pts.items() if not ext)
        dashed_ns = sorted(n for n, (_, ext) in pts.items() if ext)

        kw = dict(color=style["color"], marker=style["marker"], label=style["label"])
        if solid_ns:
            ax.plot(solid_ns, [pts[n][0] for n in solid_ns], linestyle="-", **kw)
        if dashed_ns:
            ax.plot(
                dashed_ns, [pts[n][0] for n in dashed_ns],
                linestyle="--", alpha=0.6,
                # avoid double-label
                **{**kw, "label": "_nolegend_" if solid_ns else kw["label"]},
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Dense cold-start: wall time vs n")
    ax.set_xlabel("n")
    ax.set_ylabel("wall time (s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: sparse_cold.png
# ---------------------------------------------------------------------------

def fig_sparse_cold(cells: list, figures_dir: Path) -> None:
    out_path = figures_dir / "sparse_cold.png"

    ot_map: dict[tuple, float] = {}
    pot_map: dict[tuple, float] = {}

    for c in cells:
        if c["scenario"] != "sparse_cold" or c["extrapolated"] or c["wall_s"] is None:
            continue
        key = (c["n"], c["k"])
        if c["solver"] == "sparse_ot":
            ot_map[key] = c["wall_s"]
        elif c["solver"] == "pot":
            pot_map[key] = c["wall_s"]

    common = sorted(set(ot_map) & set(pot_map))

    if not common:
        _placeholder(out_path, "No overlapping (n, k) for sparse_ot and pot")
        return

    ns = sorted({p[0] for p in common})
    ks = sorted({p[1] for p in common})
    grid = np.full((len(ns), len(ks)), np.nan)

    n_idx = {n: i for i, n in enumerate(ns)}
    k_idx = {k: j for j, k in enumerate(ks)}

    with np.errstate(divide="ignore", invalid="ignore"):
        for (n, k) in common:
            grid[n_idx[n], k_idx[k]] = np.log10(ot_map[(n, k)] / pot_map[(n, k)])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(sparse-ot / POT)")

    try:
        ax.contour(grid, levels=[0.0], colors="black", linewidths=1.5)
    except Exception:
        pass

    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_ylabel("n")
    ax.set_xlabel("k")
    ax.set_title("Sparse cold: log10(sparse-ot / POT) wall time (blue = sparse-ot faster)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: warm_speedup.png
# ---------------------------------------------------------------------------

def fig_warm_speedup(cells: list, figures_dir: Path) -> None:
    out_path = figures_dir / "warm_speedup.png"

    cold_map: dict[tuple, float] = {}
    warm_map: dict[tuple, float] = {}

    for c in cells:
        if c["extrapolated"] or c["wall_s"] is None or c["solver"] != "sparse_ot":
            continue
        key = (c["n"], c["k"])
        if c["scenario"] == "sparse_cold":
            cold_map[key] = c["wall_s"]
        elif c["scenario"] == "sparse_warm" and c.get("warm_ratio") == 0.25:
            warm_map[key] = c["wall_s"]

    common_ks = {p[1] for p in cold_map} & {p[1] for p in warm_map}

    if not common_ks:
        _placeholder(out_path, "No common k values for cold vs warm (warm_ratio=0.25)")
        return

    k_plot = min(common_ks)

    cold_pts = sorted((n, t) for (n, k), t in cold_map.items() if k == k_plot)
    warm_pts = sorted((n, t) for (n, k), t in warm_map.items() if k == k_plot)

    fig, ax = plt.subplots(figsize=(7, 5))
    if cold_pts:
        ns_c, ts_c = zip(*cold_pts)
        ax.plot(ns_c, ts_c, color="tab:blue", marker="o", linestyle="-", label="cold")
    if warm_pts:
        ns_w, ts_w = zip(*warm_pts)
        ax.plot(
            ns_w, ts_w, color="tab:orange", marker="s", linestyle="-",
            label=f"warm (ratio=0.25, k={k_plot})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Warm-start speedup at k={k_plot} (warm_ratio=0.25)")
    ax.set_xlabel("n")
    ax.set_ylabel("wall time (s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: accuracy.png
# ---------------------------------------------------------------------------

def fig_accuracy(cells: list, figures_dir: Path) -> None:
    out_path = figures_dir / "accuracy.png"

    ot_cost: dict[tuple, float] = {}
    pot_cost: dict[tuple, float] = {}

    for c in cells:
        if c["extrapolated"] or c["cost"] is None:
            continue
        key = (c["scenario"], c["n"], c["k"])
        if c["solver"] == "sparse_ot":
            ot_cost[key] = c["cost"]
        elif c["solver"] == "pot":
            pot_cost[key] = c["cost"]

    common_keys = sorted(set(ot_cost) & set(pot_cost))

    xs, ys, colors = [], [], []
    for key in common_keys:
        co = ot_cost[key]
        cp = pot_cost[key]
        rel_err = abs(co - cp) / max(abs(cp), 1e-30)
        xs.append(key[1])  # n
        ys.append(rel_err)
        colors.append("tab:blue" if key[0] == "dense_cold" else "tab:orange")

    fig, ax = plt.subplots(figsize=(7, 5))
    if xs:
        ax.scatter(xs, ys, c=colors, alpha=0.7, s=40)
        ax.axhline(1e-10, color="k", linestyle="--", label="1e-10 reference")
    else:
        _placeholder(out_path, "No common (scenario, n, k) for sparse_ot and pot costs")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("|cost_sparse_ot − cost_pot| / cost_pot")
    ax.set_title("Correctness: sparse-ot vs POT (blue=dense, orange=sparse)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def print_tables(cells: list) -> None:
    # --- Table 1: Dense cold ---
    print("\n## Dense cold-start wall time\n")
    print("| n | sparse_ot | POT | OR-Tools |")
    print("|--:|----------:|----:|---------:|")

    dense = [c for c in cells if c["scenario"] == "dense_cold"]
    ns_dense = sorted({c["n"] for c in dense})

    for n in ns_dense:
        row_cells = {c["solver"]: c for c in dense if c["n"] == n}
        parts = [str(n)]
        for solver in ("sparse_ot", "pot", "ortools"):
            if solver in row_cells:
                c = row_cells[solver]
                prefix = "~" if c["extrapolated"] else ""
                parts.append(prefix + _fmt_time(c["wall_s"]))
            else:
                parts.append("—")
        print("| " + " | ".join(parts) + " |")

    # --- Table 2: Warm-start speedup ---
    print("\n## Warm-start speedup (warm_ratio=0.25)\n")
    print("| n | k | cold | warm | speedup |")
    print("|--:|--:|-----:|-----:|--------:|")

    cold_cells = {
        (c["n"], c["k"]): c
        for c in cells
        if c["scenario"] == "sparse_cold" and c["solver"] == "sparse_ot" and not c["extrapolated"]
    }
    warm_cells = {
        (c["n"], c["k"]): c
        for c in cells
        if c["scenario"] == "sparse_warm" and c["solver"] == "sparse_ot"
        and c.get("warm_ratio") == 0.25 and not c["extrapolated"]
    }

    common = sorted(set(cold_cells) & set(warm_cells))
    for key in common:
        n, k = key
        t_cold = cold_cells[key]["wall_s"]
        t_warm = warm_cells[key]["wall_s"]
        if t_cold is not None and t_warm is not None and t_warm > 0:
            speedup = f"{t_cold / t_warm:.2f}x"
        else:
            speedup = "—"
        print(f"| {n} | {k} | {_fmt_time(t_cold)} | {_fmt_time(t_warm)} | {speedup} |")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benchmark figures from bench_<tag>.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mid", action="store_true")
    ap.add_argument("--print-tables", action="store_true")
    args = ap.parse_args()

    tag = "quick" if args.quick else ("mid" if args.mid else "full")
    data = _load(tag)
    cells = data["cells"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_dense_cold(cells, FIGURES_DIR)
    fig_sparse_cold(cells, FIGURES_DIR)
    fig_warm_speedup(cells, FIGURES_DIR)
    fig_accuracy(cells, FIGURES_DIR)

    print(f"figures in: {FIGURES_DIR}")

    if args.print_tables:
        print_tables(cells)


if __name__ == "__main__":
    main()
