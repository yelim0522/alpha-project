"""Closed-loop turn-boundary robustness audit; raw seed rows are retained.

python3 turn_validation.py --seeds 3 --users 192 --seconds 600
Outputs per-seed CSV, mean/stdev CSV, and a reproducibility manifest.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from policies import ReactiveHybrid, PallasApprox, Coordinated, TurnBoundary
from run import build_parser, build_servers, finalize_cfg, precompute_trace, run_policy


SCENARIOS = {
    "fixed": dict(),
    "variable": dict(turn_distribution="lognormal", think_distribution="mixture"),
    "azure-fixed": dict(empirical=True),
    "azure-mixture": dict(empirical=True, think_distribution="mixture"),
    "azure-decode240": dict(empirical=True, think_distribution="mixture", decode_capacity=240.0),
    "azure-dt025": dict(empirical=True, think_distribution="mixture", dt=0.25),
    "immediate": dict(think_time=0.0),
}
POLICIES = {p.name: p for p in (ReactiveHybrid, PallasApprox, Coordinated, TurnBoundary)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--users", type=int, default=192)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--workload", default=str(Path(__file__).parent / "data/azure_output_histogram.json"))
    ap.add_argument("--output", default="turn_validation_v09")
    args = ap.parse_args()
    if args.seeds <= 0 or args.users <= 0 or args.seconds <= 0:
        ap.error("seeds/users/seconds must be positive")
    scenarios, policies = args.scenarios.split(","), args.policies.split(",")
    if set(scenarios) - SCENARIOS.keys() or set(policies) - POLICIES.keys():
        ap.error("unknown scenario or policy")
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows, configs = [], {}
    with open(str(prefix) + "_seeds.csv", "w", newline="") as fh:
        writer = None
        # Cache only the two mobility grids for one seed at a time. Conversation
        # RNGs live in ConversationClock and never influence mobility draws.
        for seed in range(args.base_seed, args.base_seed + args.seeds):
            traces = {}
            for scenario in scenarios:
                cfg = finalize_cfg(build_parser().parse_args([
                    "--conversation-clock", "closed-loop", "--users", str(args.users)]))
                for key, value in SCENARIOS[scenario].items():
                    if key == "empirical":
                        cfg.turn_workload = args.workload
                    else:
                        setattr(cfg, key, value)
                cfg.seed = seed
                cfg.steps = round(args.seconds / cfg.dt)
                cfg.pred_step = cfg.dt
                configs[scenario] = vars(cfg).copy()
                if cfg.dt not in traces:
                    # Fix the reference conversation even in the immediate
                    # sensitivity. Only its initial context is used by the clock.
                    import copy
                    ref = copy.copy(cfg)
                    ref.think_time, ref.turn_output_tokens = 4.0, 128.0
                    servers = build_servers(ref, random.Random(seed + 999))
                    traces[cfg.dt] = servers, precompute_trace(ref, servers)
                servers, trace = traces[cfg.dt]
                for name in policies:
                    result = run_policy(POLICIES[name](), cfg, servers, trace)
                    row = dict(scenario=scenario, seed=seed, policy=name, **result)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)
                    fh.flush()
                    print(f"seed={seed} {scenario:16} {name:14} "
                          f"response wait={result['response_wait_mean_s']:.3f}/{result['response_wait_p99_s']:.3f}s "
                          f"done={result['responses_completed']:.0f} "
                          f"observed wait={result['observed_wait_s']:.1f}s", flush=True)
    numeric = [k for k in rows[0] if k not in ("scenario", "seed", "policy")]
    summaries = []
    for scenario in scenarios:
        for policy in policies:
            subset = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            summary = dict(scenario=scenario, policy=policy, seeds=len(subset))
            for key in numeric:
                values = [r[key] for r in subset]
                summary[key + "_mean"] = statistics.mean(values)
                summary[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summaries.append(summary)
    with open(str(prefix) + "_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    # Source hashes identify uncommitted implementations without claiming the
    # previous Git commit already contains this experiment.
    root = Path(__file__).parent
    manifest = {
        "argv": sys.argv, "seeds": list(range(args.base_seed, args.base_seed + args.seeds)),
        "configs": configs,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(root.glob("*.py"))},
        "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest()
                              if any(SCENARIOS[s].get("empirical") for s in scenarios) else None,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metric_note": "response metrics cover fully observed completed responses; censored counts and all observed wait are separate",
        "resource_note": "decode240 is an uncalibrated sensitivity budget; prefill/decode budgets are separate; resident KV is accounting, not memory admission",
    }
    Path(str(prefix) + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
