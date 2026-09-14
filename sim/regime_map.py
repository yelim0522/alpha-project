"""Regime map: which handover policy wins where.

Answers "is proactive (Pallas-style) migration always better than reactive?" by
sweeping two scenario axes at a time and reporting, per cell, the mean SIT of the
reactive hybrid (ctHO approx), uncoordinated proactive (Pallas approx) and the
coordinated planner, plus the detour alternative's ITL penalty.

Maps
  A  context length  x  users          (how much state there is to move  vs  herding)
  B  prediction noise x users          (how often preparation is wasted  vs  herding)
  C  backhaul bandwidth x context      (when copying beats recomputing)

Run:  python3 regime_map.py [--seeds 2] [--maps ABC] [--csv out.csv]
"""

import argparse
import csv
import random
from typing import Dict, List, Tuple

from policies import ReactiveHybrid, Detour, PallasApprox, Coordinated
from run import build_parser, finalize_cfg, build_servers, precompute_trace, run_policy, average

POLICIES = {"R": ReactiveHybrid, "P": PallasApprox, "C": Coordinated, "D": Detour}
LEGEND = "R = reactive ctHO approx, P = Pallas approx (uncoordinated proactive), C = coordinated proactive"

# Axis definitions: (label, dict of cfg overrides)
CONTEXT_BANDS = [("0.5K", dict(min_context=125.0, max_context=500.0)),
                 ("1K", dict(min_context=250.0, max_context=1000.0)),
                 ("2K", dict(min_context=500.0, max_context=2000.0)),
                 ("4.5K", dict(min_context=500.0, max_context=4500.0)),
                 ("8K", dict(min_context=2000.0, max_context=8000.0))]
USER_COUNTS = [("32", dict(users=32)), ("64", dict(users=64)),
               ("128", dict(users=128)), ("192", dict(users=192))]
NOISE_LEVELS = [("none", dict(pred_speed_noise=0.0, pred_heading_noise=0.0)),
                ("low", dict(pred_speed_noise=0.15, pred_heading_noise=0.2)),
                ("mid", dict(pred_speed_noise=0.30, pred_heading_noise=0.4)),
                ("high", dict(pred_speed_noise=0.50, pred_heading_noise=0.7))]
BACKHAUL = [("100M", dict(backhaul_mbps=100.0)), ("300M", dict(backhaul_mbps=300.0)),
            ("1G", dict(backhaul_mbps=1000.0)), ("3G", dict(backhaul_mbps=3000.0))]

MAPS = {
    "A": ("context (rows) x users (cols)", CONTEXT_BANDS, USER_COUNTS),
    "B": ("prediction noise (rows) x users (cols)", NOISE_LEVELS, USER_COUNTS),
    "C": ("backhaul (rows) x context (cols)", BACKHAUL, CONTEXT_BANDS),
}


def base_cfg(argv: List[str]):
    """Default run.py configuration, with min/max context left for the sweep to set."""
    cfg = build_parser().parse_args(argv)
    cfg.min_context = cfg.max_context = None
    return cfg


def run_cell(cfg, seeds: int) -> Dict[str, dict]:
    out: Dict[str, List[dict]] = {k: [] for k in POLICIES}
    for k in range(seeds):
        cfg.seed = cfg.base_seed + k
        servers = build_servers(cfg, random.Random(cfg.seed + 999))
        trace = precompute_trace(cfg, servers)
        for key, cls in POLICIES.items():
            out[key].append(run_policy(cls(), cfg, servers, trace))
    return {k: average(v) for k, v in out.items()}


def winner(cell: Dict[str, dict], metric: str, tie: float = 0.03) -> str:
    vals = {k: cell[k][metric] for k in ("R", "P", "C")}
    best = min(vals.values())
    close = [k for k, v in vals.items() if v <= best * (1 + tie) + 1e-9]
    return "=".join(sorted(close, key=lambda k: vals[k]))


def fmt_cell(cell: Dict[str, dict]) -> str:
    r, p, c = (cell[k]["sit_mean_s"] for k in ("R", "P", "C"))
    return f"{r:.2f}/{p:.2f}/{c:.2f}"


def run_map(name: str, argv: List[str], seeds: int, writer=None):
    title, rows, cols = MAPS[name]
    print(f"Map {name}: {title}.  Cell = mean SIT (s) R/P/C; [w] = winner by mean SIT, "
          f"[w99] by p99 (ties within 3%).")
    head = f"{'':>8}" + "".join(f"{c:>26}" for c, _ in cols)
    print(head)
    ratios_pr, ratios_cp = [], []
    for rlab, rov in rows:
        line = f"{rlab:>8}"
        for clab, cov in cols:
            cfg = base_cfg(argv)
            for k, v in {**rov, **cov}.items():
                setattr(cfg, k, v)
            finalize_cfg(cfg)
            cell = run_cell(cfg, seeds)
            w, w99 = winner(cell, "sit_mean_s"), winner(cell, "sit_p99_s")
            line += f"{fmt_cell(cell):>17} [{w:>3}|{w99:>3}]"
            r, p, c = (cell[k]["sit_mean_s"] for k in ("R", "P", "C"))
            ratios_pr.append((rlab, clab, p / r if r > 0 else float("nan")))
            ratios_cp.append((rlab, clab, c / p if p > 0 else float("nan")))
            if writer:
                base_itl = cell["R"]["itl_ms"]
                writer.writerow([name, rlab, clab, f"{r:.4f}", f"{p:.4f}", f"{c:.4f}",
                                 f"{cell['R']['sit_p99_s']:.4f}", f"{cell['P']['sit_p99_s']:.4f}",
                                 f"{cell['C']['sit_p99_s']:.4f}",
                                 f"{cell['P']['prep_rate_pct']:.0f}", f"{cell['C']['prep_rate_pct']:.0f}",
                                 f"{cell['P']['wasted_mb']:.0f}", f"{cell['C']['wasted_mb']:.0f}",
                                 f"{cell['P']['peak_streams']:.0f}", f"{cell['C']['peak_streams']:.0f}",
                                 f"{cell['D']['itl_ms'] - base_itl:.1f}", w, w99])
        print(line)
    print()
    _ratio_grid("Pallas / reactive  (mean SIT; <1 = proactive helps, >1 = proactive hurts)",
                rows, cols, ratios_pr)
    _ratio_grid("coordinated / Pallas  (mean SIT; <1 = coordination helps)", rows, cols, ratios_cp)


def _ratio_grid(title: str, rows, cols, ratios: List[Tuple[str, str, float]]):
    print(f"  {title}")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c, _ in cols))
    lookup = {(r, c): v for r, c, v in ratios}
    for rlab, _ in rows:
        print(f"{rlab:>8}" + "".join(f"{lookup[(rlab, clab)]:>8.2f}" for clab, _ in cols))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--maps", default="ABC", help="subset of A, B, C")
    ap.add_argument("--csv", default="", help="write per-cell numbers to this CSV")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--model", default="qwen32b")
    a = ap.parse_args()
    argv = ["--steps", str(a.steps), "--model", a.model]
    print(LEGEND)
    print(f"base: run.py defaults (6 servers, GroupFlow platoon 8, 25 m/s, {a.steps}x0.5 s, "
          f"model={a.model}, P=4, VRAM 6 GB), seeds={a.seeds}\n")
    writer, fh = None, None
    if a.csv:
        fh = open(a.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["map", "row", "col", "sit_R", "sit_P", "sit_C", "p99_R", "p99_P", "p99_C",
                         "prep_P", "prep_C", "waste_P_mb", "waste_C_mb", "peak_P", "peak_C",
                         "detour_itl_penalty_ms", "winner_mean", "winner_p99"])
    for m in a.maps:
        run_map(m, argv, a.seeds, writer)
    if fh:
        fh.close()
    print("Detour is omitted from the winner because its SIT is 0 by construction; its cost is the "
          "ITL penalty (detour_itl_penalty_ms in the CSV).")


if __name__ == "__main__":
    main()
