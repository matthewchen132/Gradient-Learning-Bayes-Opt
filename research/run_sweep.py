"""
Run the Monte Carlo sweep defined by config.SweepConfig and collect results.

    python run_sweep.py                       # full sweep, 80% of CPUs (override with --workers N)
    python run_sweep.py --dry-run             # list runs only
    python run_sweep.py --max-runs 8 --end-time 1.0   # smoke test

Output: <repo>/outputs/sweep_<YYYY-mm-dd_HHMMSS>/{index.csv, <run_name>/{config.json, steps.npz, summary.json}}
Metric definitions: metrics_to_track.md
"""

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from config import SweepConfig

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def _flatten(d, prefix=""):
    """{'gibo': {'M': 2}} → {'gibo.M': 2}; tuples become strings for CSV."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v) if isinstance(v, (tuple, list)) else v
    return out


def run_one(args):
    """Worker: one run → (run_name, summary). Never raises; failures are recorded in the summary."""
    cfg, run_dir = args
    import torch
    torch.set_num_threads(1)                         # one BLAS thread per process — avoids oversubscription
    import GIBO_Robotic_Manipulator as G             # sets float64 default dtype on import
    from src.core.logger import RunLogger

    logger = RunLogger(cfg, out_dir=run_dir)         # writes config.json immediately
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # silence per-step / solver prints
            summary = G.main(cfg, logger)
    except Exception as e:
        logger.status = f"failed: {type(e).__name__}: {e}"
        summary = logger.save()                      # partial steps.npz + summary.json
    return cfg.run_name(), {**_flatten(cfg.to_dict()), **summary}


def write_index(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))   # union of keys, first-seen order
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

HEADLINE = ("n_violations", "min_h", "grad_err_mean", "grad_std_auc", "grad_coverage",
            "h_std_mean", "h_coverage", "excess_gap", "intervention_rate", "runtime_total")
ALGO_ORDER = {"oracle": 0, "gibo": 1, "random": 2, "lambda": 3}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None          # "", "None", text → not numeric


def summarize_sweep(sweep_dir):
    """
    Average every numeric summary metric over θ₀ and seeds, grouped by (algo, noise_std, M).
    Adds excess_gap = boundary_gap − boundary_gap of the matching oracle run (same noise, θ₀, seed).
    Writes <sweep_dir>/summary_by_algo.csv (mean and std per metric) and prints a headline table.
    """
    from config import RunConfig
    sweep_dir = Path(sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "index.csv")))
    ok   = [r for r in rows if r["status"].startswith("ok")]

    # -- Excess gap vs. the matching oracle run --
    oracle_gap = {(r["noise_std"], r["theta0_id"], r["seed"]): _num(r["boundary_gap"])
                  for r in ok if r["algo"] == "oracle"}
    for r in ok:
        gap, base = _num(r["boundary_gap"]), oracle_gap.get((r["noise_std"], r["theta0_id"], r["seed"]))
        r["excess_gap"] = gap - base if gap is not None and base is not None else ""

    # -- Metric columns = everything that isn't a config field --
    config_cols = set(_flatten(RunConfig().to_dict()))
    metrics     = [k for k in ok[0] if k not in config_cols and k != "status"]

    # -- Group by (algo, noise, M); mean / std over θ₀ × seeds --
    def key(r):
        return (r["algo"], float(r["noise_std"]), int(r["gibo.M"]))
    groups = {}
    for r in rows:
        groups.setdefault(key(r), {"ok": [], "n_failed": 0})
        if r["status"].startswith("ok"):
            groups[key(r)]["ok"].append(r)
        else:
            groups[key(r)]["n_failed"] += 1

    out = []
    for (algo, noise, M) in sorted(groups, key=lambda g: (g[1], g[2], ALGO_ORDER.get(g[0], 9))):
        g   = groups[(algo, noise, M)]
        row = {"algo": algo, "noise_std": noise, "M": M, "n_runs": len(g["ok"]), "n_failed": g["n_failed"]}
        for m in metrics:
            vals = np.array([v for v in (_num(r.get(m)) for r in g["ok"]) if v is not None])
            row[f"{m}_mean"] = float(vals.mean()) if vals.size else ""
            row[f"{m}_std"]  = float(vals.std())  if vals.size else ""
        out.append(row)
    write_index(sweep_dir / "summary_by_algo.csv", out)

    # -- Headline table --
    print(f"\n{'algo':7s} {'noise':>5s} {'M':>2s} {'n':>3s} " + " ".join(f"{m[:14]:>16s}" for m in HEADLINE))
    for row in out:
        cells = [f"{row[m + '_mean']:7.3f}±{row[m + '_std']:<7.3f}" if row.get(m + "_mean") != "" else f"{'—':>16s}"
                 for m in HEADLINE]
        print(f"{row['algo']:7s} {row['noise_std']:5.2f} {row['M']:2d} {row['n_runs']:3d} " + " ".join(cells))
    print(f"\n→ {sweep_dir / 'summary_by_algo.csv'}")
    return out


PLOT_METRICS = (    # (summary column, panel title, y-axis label) — one per metric category
    ("n_violations",  "Safety violations",          "steps with h < 0"),
    ("grad_err_mean", "Gradient error",             "mean ‖μ_∇ − ∇h‖"),
    ("grad_std_auc",  "Gradient uncertainty (AUC)", "∫ √tr Σ_∇ dt"),
    ("h_std_mean",    "h uncertainty",              "mean h_std [rad]"),
    ("excess_gap",    "Excess gap vs. oracle",      "boundary gap − oracle [rad]"),
    ("runtime_total", "Runtime",                    "seconds per run"),
)
PLOT_ALGOS  = ("gibo", "random", "lambda")
ALGO_LABELS = {"gibo": "vanilla GIBO", "random": "random", "lambda": "λ-weighted"}
ALGO_COLORS = {"gibo": "#2a78d6", "random": "#eb6834", "lambda": "#1baf7a"}   # categorical slots 1–3 (CVD-validated)
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"


def plot_metric_comparison(sweep_dir, summary, show=True):
    """2×3 bar grid: one panel per PLOT_METRICS entry; bars = algorithms, groups = (noise, M) cells, error bars = ±1 std."""
    import matplotlib
    import matplotlib.pyplot as plt

    sweep_dir = Path(sweep_dir)
    by_key = {(r["algo"], float(r["noise_std"]), int(r["M"])): r for r in summary}
    cells  = sorted({(n, M) for (a, n, M) in by_key if a in PLOT_ALGOS})
    x, w   = np.arange(len(cells)), 0.26

    def series(algo, col):
        vals = [_num(by_key.get((algo, *c), {}).get(col)) for c in cells]
        return np.array([np.nan if v is None else v for v in vals])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for ax, (col, title, ylabel) in zip(axes.flat, PLOT_METRICS):
        ax.set_facecolor(SURFACE)
        for i, algo in enumerate(PLOT_ALGOS):
            mean, std = series(algo, col + "_mean"), series(algo, col + "_std")
            lower = std if col == "excess_gap" else np.minimum(std, mean)   # non-negative metrics: don't draw error below 0
            ax.bar(x + (i - 1) * w, mean, w, yerr=[lower, std],
                   color=ALGO_COLORS[algo], edgecolor=SURFACE, linewidth=1.5, label=ALGO_LABELS[algo],
                   error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2), zorder=2)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_xticks(x, [f"σ={n:g}\nM={M}" for n, M in cells])
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Acquisition comparison — mean ± 1 std over ~15 runs per bar (5 θ₀ × 3 seeds); lower is better",
                 color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    path = sweep_dir / "metric_comparison.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"→ {path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path


def plot_run(run_dir):
    """
    Create a plot from `steps.npz` and `config.json` (no re-simulation) and save it as
    <run_dir>/<run_dir.name>.png. 
    
    Tracks the following: 
      θ₁ | h uncertainty | gradient uncertainty.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    if not (run_dir / "steps.npz").exists():
        return None                                       # run crashed before its first step
    cfg     = json.loads((run_dir / "config.json").read_text())
    status  = json.loads((run_dir / "summary.json").read_text()).get("status", "?") if (run_dir / "summary.json").exists() else "?"
    a       = np.load(run_dir / "steps.npz")
    algo    = cfg["algo"]                                 # Distinguish algorithms
    color   = ALGO_COLORS.get(algo, INK)
    n_sig   = cfg["gp"]["n_sigma"]
    t, th1  = a["t"], np.degrees(a["theta"][:, 0])

    fig, axes = plt.subplots(1, 3 if algo != "oracle" else 1, figsize=(17, 4.6) if algo != "oracle" else (7, 4.6),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel("time [s]", color=INK_2, fontsize=9)

    # (a) θ₁ tracking, ±nσ GP band, query points, safety limit
    ax = axes[0]
    ax.plot(t, np.degrees(a["ref"][:, 0]), "--", color=INK_2, lw=1.2, label="reference")
    if algo != "oracle":
        band = np.degrees(a["h_std"]) * n_sig
        ax.fill_between(t, th1 - band, th1 + band, color=color, alpha=0.18, lw=0, label=f"±{n_sig:g}σ GP")
        q = a["queries"]                                   # (T, M, 2)
        if q.size:
            ax.scatter(np.repeat(t, q.shape[1]), np.degrees(q[:, :, 0].ravel()), s=7, color=INK_2,
                       alpha=0.3, lw=0, label="query points", zorder=1)
    ax.plot(t, th1, color=color, lw=2, label=ALGO_LABELS.get(algo, algo), zorder=3)
    ax.axhline(cfg["cbf"]["THETA_MAX_deg"], color="#e34948", ls="--", lw=1.2, label="θ_max")
    ax.set_title("Joint 1 angle", loc="left", color=INK, fontsize=11)
    ax.set_ylabel("θ₁ [deg]", color=INK_2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    if algo != "oracle":
        # (b) value uncertainty at θ_t
        axes[1].plot(t, a["h_std"], color=color, lw=2)
        axes[1].set_title("h uncertainty at θ_t", loc="left", color=INK, fontsize=11)
        axes[1].set_ylabel("posterior std of h [rad]", color=INK_2, fontsize=9)
        axes[1].set_ylim(bottom=0)
        # (c) gradient uncertainty at θ_t
        axes[2].plot(t, a["grad_std"], color=color, lw=2)
        axes[2].set_title("Gradient uncertainty at θ_t", loc="left", color=INK, fontsize=11)
        axes[2].set_ylabel("√tr Σ_∇", color=INK_2, fontsize=9)
        axes[2].set_ylim(bottom=0)

    lam_txt = ""
    if algo == "lambda" and a["acq_value"].size:
        lam = a["acq_value"][:, 0]
        lam_txt = f",  λ mean {lam.mean():.2f} [{lam.min():.2f}, {lam.max():.2f}]"
    fail_txt = "" if status.startswith("ok") else f"   — {status[:60]} (t = {t[-1]:.2f} s)"
    fig.suptitle(f"{ALGO_LABELS.get(algo, algo)} — {cfg['optim_type']}, M={cfg['gibo']['M']}, σₙ={cfg['noise_std']:g}, "
                 f"θ₀ #{cfg['theta0_id']}, seed {cfg['seed']}{lam_txt}{fail_txt}", color=INK, fontsize=11)
    fig.tight_layout()
    path = run_dir / f"{run_dir.name}.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)                                         # free memory — hundreds of figures per sweep
    return path


def plot_runs(sweep_dir, workers=None):
    """Save plot_run() for every run folder in a sweep (parallel; existing PNGs are overwritten)."""
    run_dirs = sorted(p for p in Path(sweep_dir).iterdir() if (p / "config.json").exists())
    workers  = workers or max(1, (os.cpu_count() or 2) * 4 // 5)
    print(f"Plotting {len(run_dirs)} runs → {sweep_dir}  (workers={workers})")
    done = 0
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=50) as pool:
        for path in pool.imap_unordered(plot_run, run_dirs):
            done += path is not None
    print(f"Saved {done} per-run plots ({len(run_dirs) - done} skipped: no steps logged)")


def load_time_series(sweep_dir, field):
    """Scrape one steps.npz field from every run → (t, {(algo, noise, M): (n_runs, T) array}). Failed (shorter) runs dropped."""
    data, t = {}, None
    for f in Path(sweep_dir).glob("*/steps.npz"):
        algo, _, M, noise = f.parent.name.split("_")[:4]          # e.g. gibo_SOCP_M2_n0.5_th0_s0
        a = np.load(f)
        if field not in a.files:                                   # e.g. grad_err in sweeps from before it was logged
            continue
        data.setdefault((algo, float(noise[1:]), int(M[1:])), []).append(a[field])
        t = a["t"] if t is None or len(a["t"]) > len(t) else t
    return t, {k: np.stack([y for y in v if len(y) == len(t)]) for k, v in data.items()}


def plot_uncertainty_over_time(sweep_dir):
    """Median (line) and 25–75th percentile (band) of grad_std and h_std over time per algorithm → <sweep_dir>/summary/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(sweep_dir) / "summary"
    out_dir.mkdir(exist_ok=True)
    for field, ylabel in (("grad_std", "gradient uncertainty √tr Σ_∇"),
                          ("grad_err", "gradient error ‖μ_∇ − ∇h‖"),
                          ("h_std",    "h uncertainty [rad]")):
        t, data = load_time_series(sweep_dir, field)
        if not data:
            print(f"  skip {field}_over_time.png — '{field}' not logged in this sweep")
            continue
        noises = sorted({n for (a, n, M) in data if a in PLOT_ALGOS})
        Ms     = sorted({M for (a, n, M) in data if a in PLOT_ALGOS})
        fig, axes = plt.subplots(len(noises), len(Ms), figsize=(6.5 * len(Ms), 3 * len(noises)), sharex=True, squeeze=False)
        for (i, noise), (j, M) in ((a, b) for a in enumerate(noises) for b in enumerate(Ms)):
            ax, top = axes[i, j], 0.0
            for algo in PLOT_ALGOS:
                q25, med, q75 = np.percentile(data[(algo, noise, M)], [25, 50, 75], axis=0)
                ax.plot(t, med, color=ALGO_COLORS[algo], lw=2, label=ALGO_LABELS[algo])
                ax.fill_between(t, q25, q75, color=ALGO_COLORS[algo], alpha=0.2, lw=0)
                top = max(top, q75[5:].max())                         # ignore the t≈0 spike for the y-range
            ax.set_ylim(0, 1.1 * top)
            ax.set_title(f"σₙ = {noise:g}, M = {M}", loc="left", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)               # left column only — no overlap with tick labels
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.supxlabel("time [s]")
        fig.suptitle(f"{ylabel} at θ_t — median (line), 25–75th percentile (band) over completed runs; lower is better",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{field}_over_time.png", dpi=150)
        plt.close(fig)
        print(f"→ {out_dir / f'{field}_over_time.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) * 4 // 5))   # 80% of cores, rounded down (8 on this Mac): ~3.2 GB peak
    p.add_argument("--max-runs", type=int, default=None, help="only the first N runs (after ordering)")
    p.add_argument("--end-time", type=float, default=None, help="override sim end_time (smoke tests)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-runs", metavar="SWEEP_DIR", default=None,
                   help="only saves run plots for EXISTING sweep folders, then exit")
    p.add_argument("--time-plots", metavar="SWEEP_DIR", default=None,
                   help="only save the median-over-time plots (summary/) for an EXISTING sweep folder, then exit")
    args = p.parse_args()

    if args.plot_runs:
        plot_runs(args.plot_runs, workers=args.workers)
        return
    if args.time_plots:
        plot_uncertainty_over_time(args.time_plots)
        return

    runs = list(SweepConfig().runs())
    if args.end_time is not None:
        runs = [replace(c, sim=replace(c.sim, end_time=args.end_time)) for c in runs]
    runs.sort(key=lambda c: -c.gibo.M)               # longest first → better load balancing
    if args.max_runs:
        # interleave algorithms so a short smoke test covers all of them
        by_algo = {}
        for c in runs:
            by_algo.setdefault(c.algo, []).append(c)
        runs = [c for group in zip(*by_algo.values()) for c in group][:args.max_runs]

    sweep_dir = OUTPUT_ROOT / f"sweep_{datetime.now():%Y-%m-%d_%H%M%S}"
    print(f"{len(runs)} runs → {sweep_dir}  (workers={args.workers})")
    if args.dry_run:
        for c in runs:
            print("  ", c.run_name())
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, sweep_dir / c.run_name()) for c in runs]
    rows, t0 = [], time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:   # fresh process per run: no memory build-up
        for i, (name, row) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            write_index(sweep_dir / "index.csv", rows)          # crash-safe: rewritten after every run
            elapsed = time.perf_counter() - t0
            eta = timedelta(seconds=int(elapsed / i * (len(jobs) - i)))
            print(f"[{i}/{len(jobs)}] {name:40s} {row['status'][:40]:10s} {row['runtime_total']:6.1f}s  ETA {eta}", flush=True)

    n_fail = sum(not r["status"].startswith("ok") for r in rows)
    print(f"Done in {timedelta(seconds=int(time.perf_counter() - t0))} — {len(rows) - n_fail} ok, {n_fail} failed → {sweep_dir / 'index.csv'}")

    # -- Average over θ₀ × seeds per (algo, noise, M), then optionally plot --
    summary = summarize_sweep(sweep_dir)
    if not sys.stdin.isatty():                       # background / piped run: no one to ask, save the plot anyway
        plot_metric_comparison(sweep_dir, summary, show=False)
    elif input("Plot metric comparisons across algorithms? [Y/n] ").strip().lower() in ("", "y", "yes"):
        plot_metric_comparison(sweep_dir, summary)

    # -- Per-run figures, rebuilt from each steps.npz (no re-simulation) --
    if not sys.stdin.isatty() or input("Save per-run plots? [Y/n] ").strip().lower() in ("", "y", "yes"):
        plot_runs(sweep_dir, workers=args.workers)

    # -- Median + 25–75th percentile band over time, per algorithm → summary/ --
    if not sys.stdin.isatty() or input("Plot median uncertainty over time? [Y/n] ").strip().lower() in ("", "y", "yes"):
        plot_uncertainty_over_time(sweep_dir)


if __name__ == "__main__":
    main()
