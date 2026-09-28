"""Small reserved-v1 correctness audit, NOT a final performance comparison.

From repository root: python3 sim/validate_memory.py
Writes a new artifact with per-seed metrics, config and implementation hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from memory import PreparationMemory
from policies import PallasApprox, Coordinated, HedgedPallas
from run import build_parser, finalize_cfg, build_servers, precompute_trace, run_policy


SCENARIOS = {
    'trace-default': [],
    'trace-tight': ['--vram-mb', '600'],
    'closed-tight': ['--vram-mb', '600', '--conversation-clock', 'closed-loop'],
    'closed-noisy-hedging': ['--vram-mb', '600', '--conversation-clock', 'closed-loop',
                             '--pred-speed-noise', '.2', '--pred-heading-noise', '.3',
                             '--pred-candidates', '2', '--pred-samples', '8'],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('memory_validation_v1.json'))
    args = ap.parse_args()
    rows = []
    for scenario, flags in SCENARIOS.items():
        for seed in (0, 1):
            cfg = finalize_cfg(build_parser().parse_args([
                '--memory-mode', 'reserved', '--users', '32', '--steps', '240'] + flags))
            cfg.seed = seed
            servers = build_servers(cfg, random.Random(seed + 999))
            trace = precompute_trace(cfg, servers)
            for policy in (PallasApprox(), Coordinated(detour=False, label='coordinated-core-v1'),
                           HedgedPallas()):
                metrics = run_policy(policy, cfg, servers, trace)
                assert metrics['memory_violations'] == 0
                assert metrics['memory_peak_budget_fraction'] <= 1 + 1e-10
                assert not policy.memory.entries
                assert metrics['memory_admissions'] == metrics['memory_releases']
                starts = [e for e in policy.memory.events if e['event'] == 'admit']
                rows.append(dict(scenario=scenario, seed=seed, policy=policy.name,
                                 detour_hops=getattr(policy, 'detour_hops', 0),
                                 config=vars(cfg).copy(), metrics=metrics,
                                 start_event_sample=starts[:3]))
                print(f"{scenario} seed={seed} {policy.name}: "
                      f"admitted={metrics['memory_admissions']} "
                      f"deferred={metrics['memory_defer_attempts']} "
                      f"cancelled={metrics['memory_cancels']} violations=0", flush=True)
    root = Path(__file__).resolve().parent
    result = dict(purpose='Day 2 correctness smoke audit; not final performance evidence',
                  memory_version=PreparationMemory.version, python=sys.version,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                    text=True).strip(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.glob('*.py'))}, rows=rows)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Saved {len(rows)} runs to {args.output}')


if __name__ == '__main__':
    main()
