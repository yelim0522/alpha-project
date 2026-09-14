"""Evaluate three alternative baselines that can change the paper's framing.

T: turn-boundary migration under an explicit generation/think cycle.
H: full multi-candidate Pallas hedging from a noisy trajectory ensemble.
K: ideal KV compression-ratio sensitivity (an optimistic upper bound: codec
   compute/latency is deliberately omitted).

Run: python3 alternatives.py --seeds 3 --csv alternatives.csv
"""

import argparse
import csv
import random
from typing import Dict, Iterable, List

from policies import ReactiveHybrid, PallasApprox, Coordinated, HedgedPallas, TurnBoundary
from run import average, build_parser, build_servers, finalize_cfg, precompute_trace, run_policy


NOISE = [
    ("none", 0.0, 0.0),
    ("low", 0.15, 0.2),
    ("mid", 0.30, 0.4),
    ("high", 0.50, 0.7),
]
RATIOS = [1.0, 0.5, 0.25, 0.125]
SUMMARY_KEYS = [
    "handovers", "sit_mean_s", "sit_p99_s", "sit_max_s", "itl_ms",
    "prep_rate_pct", "wasted_mb", "peak_streams", "load_cov", "jain",
    "transfer_mb", "hedge_copies", "hedge_hit_pct", "hedge_alt_hit_pct", "boundary_moves",
    "boundary_hidden_pct", "boundary_wait_s",
]


def make_cfg(args, users: int, *, turn_model=False, candidates=1, samples=1,
             speed_noise=0.0, heading_noise=0.0, compression=1.0):
    argv = ["--users", str(users), "--steps", str(args.steps), "--model", args.model,
            "--pred-speed-noise", str(speed_noise),
            "--pred-heading-noise", str(heading_noise),
            "--pred-candidates", str(candidates), "--pred-samples", str(samples),
            "--kv-compression-ratio", str(compression),
            "--turn-output-tokens", str(args.turn_output_tokens),
            "--think-time", str(args.think_time)]
    if turn_model:
        argv.append("--turn-model")
    return finalize_cfg(build_parser().parse_args(argv))


def evaluate(cfg, seeds: int, policy_factories: Iterable):
    totals: Dict[str, List[dict]] = {}
    for k in range(seeds):
        cfg.seed = cfg.base_seed + k
        servers = build_servers(cfg, random.Random(cfg.seed + 999))
        trace = precompute_trace(cfg, servers)
        for factory in policy_factories:
            policy = factory()
            totals.setdefault(policy.name, []).append(run_policy(policy, cfg, servers, trace))
    return [(name, average(values)) for name, values in totals.items()]


def print_rows(title: str, condition: str, rows):
    print(f"{title}: {condition}")
    print(f"{'policy':<18} {'SITavg':>8} {'SITp99':>8} {'ITLms':>8} "
          f"{'wasteMB':>10} {'peak':>6} {'extra':>12}")
    for name, s in rows:
        if name == "turn-boundary":
            extra = f"hide={s['boundary_hidden_pct']:.0f}%"
        elif name == "pallas-hedge2":
            extra = f"alt={s['hedge_alt_hit_pct']:.0f}%"
        else:
            extra = "-"
        print(f"{name:<18} {s['sit_mean_s']:>8.3f} {s['sit_p99_s']:>8.3f} "
              f"{s['itl_ms']:>8.1f} {s['wasted_mb']:>10.0f} "
              f"{s['peak_streams']:>6.0f} {extra:>12}")
    print()


def emit(writer, map_name, row, rows):
    if writer is None:
        return
    for policy, summary in rows:
        writer.writerow([map_name, row, policy] + [summary[k] for k in SUMMARY_KEYS])


def run_turn_map(args, writer):
    policies = [ReactiveHybrid, PallasApprox, Coordinated, TurnBoundary]
    for users in args.users:
        cfg = make_cfg(args, users, turn_model=True)
        rows = evaluate(cfg, args.seeds, policies)
        condition = (f"users={users}, output={args.turn_output_tokens:g} tokens, "
                     f"think={args.think_time:g}s")
        print_rows("Map T (turn boundary)", condition, rows)
        emit(writer, "T", f"users={users}", rows)
    for think_time in args.think_times:
        original = args.think_time
        args.think_time = think_time
        cfg = make_cfg(args, args.focus_users, turn_model=True)
        rows = evaluate(cfg, args.seeds, policies)
        condition = (f"users={args.focus_users}, output={args.turn_output_tokens:g} tokens, "
                     f"think={think_time:g}s")
        print_rows("Map T2 (think-time sensitivity)", condition, rows)
        emit(writer, "T2", f"think={think_time:g}", rows)
        args.think_time = original


def run_hedge_map(args, writer):
    policies = [PallasApprox,
                lambda: HedgedPallas(max_candidates=args.candidates),
                Coordinated]
    for label, speed_noise, heading_noise in NOISE:
        cfg = make_cfg(args, args.focus_users, candidates=args.candidates,
                       samples=args.candidate_samples, speed_noise=speed_noise,
                       heading_noise=heading_noise)
        rows = evaluate(cfg, args.seeds, policies)
        condition = (f"noise={label}, users={args.focus_users}, "
                     f"top-{args.candidates}/{args.candidate_samples} samples")
        print_rows("Map H (candidate hedging)", condition, rows)
        emit(writer, "H", f"noise={label}", rows)


def run_compression_map(args, writer):
    policies = [ReactiveHybrid, PallasApprox, Coordinated]
    for ratio in RATIOS:
        cfg = make_cfg(args, args.focus_users, compression=ratio)
        rows = evaluate(cfg, args.seeds, policies)
        condition = f"compressed/raw={ratio:g}, users={args.focus_users} (ideal codec)"
        print_rows("Map K (KV compression)", condition, rows)
        emit(writer, "K", f"ratio={ratio:g}", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default="THK", help="subset of T (turn), H (hedge), K (compression)")
    ap.add_argument("--users", default="32,64,128,192", help="density list for Map T")
    ap.add_argument("--focus-users", type=int, default=192, help="users for Maps H and K")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="qwen32b")
    ap.add_argument("--turn-output-tokens", type=float, default=128.0)
    ap.add_argument("--think-time", type=float, default=4.0)
    ap.add_argument("--think-times", default="0,1,2,4,8",
                    help="think-time sensitivity list for Map T2")
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--candidate-samples", type=int, default=8)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    args.users = [int(x) for x in args.users.split(",") if x]
    args.think_times = [float(x) for x in args.think_times.split(",") if x]

    fh = open(args.csv, "w", newline="") if args.csv else None
    writer = csv.writer(fh) if fh else None
    if writer:
        writer.writerow(["map", "condition", "policy"] + SUMMARY_KEYS)
        fh.flush()
    try:
        for name, fn in (("T", run_turn_map), ("H", run_hedge_map),
                         ("K", run_compression_map)):
            if name in args.maps.upper():
                fn(args, writer)
                if fh:
                    fh.flush()
    finally:
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
