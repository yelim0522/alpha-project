"""Entry point: builds a scenario, precomputes one shared mobility/token trace,
then replays it under every policy for a fair comparison.

Run:  python run.py
Optional flags: --users, --servers, --steps, --seed, --speed
"""

import argparse
import random

from environment import (
    Server, User, CostParams, nearest_server, distance, effective_bandwidth,
)
from mobility import RandomWaypoint
from metrics import Metrics
from policies import all_policies


def build_servers(m, width, height, coverage, prefill_speed, rng):
    """Place servers on a rough grid so coverage areas overlap (enables handovers)."""
    cols = int(m ** 0.5 + 0.999) or 1
    rows = (m + cols - 1) // cols
    servers = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= m:
                break
            x = (c + 0.5) * width / cols
            y = (r + 0.5) * height / rows
            servers.append(Server(idx, x, y, coverage, prefill_speed))
            idx += 1
    return servers


def precompute_trace(cfg, servers):
    """Produce identical per-step positions and token counts for all policies."""
    rng = random.Random(cfg.seed)
    mob = RandomWaypoint(cfg.width, cfg.height, cfg.speed, rng)
    users = [User(id=i,
                  x=rng.uniform(0, cfg.width),
                  y=rng.uniform(0, cfg.height),
                  tokens=rng.uniform(50, 400))
             for i in range(cfg.users)]

    trace = []  # trace[t] = list of (x, y, vx, vy, tokens) per user
    for _ in range(cfg.steps):
        snapshot = []
        for u in users:
            mob.step(u, cfg.dt)
            u.tokens = min(cfg.max_context, u.tokens + cfg.token_rate * cfg.dt)
            snapshot.append((u.x, u.y, u.vx, u.vy, u.tokens))
        trace.append(snapshot)
    return trace


def run_policy(policy, cfg, servers, trace):
    params = CostParams(kv_bytes_per_token=cfg.kv_bytes_per_token,
                        backhaul_bw=cfg.backhaul_bw,
                        contention_alpha=cfg.contention_alpha)
    users = [User(id=i, x=0.0, y=0.0) for i in range(cfg.users)]
    metrics = Metrics()

    for t_idx in range(cfg.steps):
        t = t_idx * cfg.dt
        snap = trace[t_idx]
        for u, (x, y, vx, vy, tokens) in zip(users, snap):
            u.x, u.y, u.vx, u.vy, u.tokens = x, y, vx, vy, tokens

        gate_level = mobility_gate(users, servers)
        metrics.prefetch_activations += policy.on_step(
            t, cfg.dt, users, servers, params, gate_level)

        # Detect handovers this step and their backhaul concurrency.
        pending = []
        for u in users:
            target = nearest_server(u, servers)
            if u.server == -1:
                u.server = target.id
                u.history.append((target.id, t))
                continue
            if target.id != u.server:
                pending.append((u, target))

        bw_eff = effective_bandwidth(params.backhaul_bw, len(pending),
                                     params.contention_alpha)
        for u, target in pending:
            is_pingpong = any(sid == target.id and t - ts <= cfg.pingpong_window
                              for sid, ts in u.history)
            delay, _, transfer_tok, used = policy.plan_handover(u, target, params, bw_eff)
            metrics.record_handover(delay, transfer_tok, is_pingpong, used)
            u.server = target.id
            u.history.append((target.id, t))
            if len(u.history) > 16:
                u.history.pop(0)

    return metrics.summary()


def mobility_gate(users, servers):
    """Fraction of users near a coverage boundary -> proxy for handover pressure.
    Returns a value in [0, 1] used to scale prefetch intensity (safety gate)."""
    if not users:
        return 0.0
    near = 0
    for u in users:
        s = nearest_server(u, servers)
        d = distance(u.x, u.y, s.x, s.y)
        if d > 0.7 * s.coverage:
            near += 1
    return near / len(users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--servers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=25.0)
    cfg = ap.parse_args()

    # Static scenario parameters (edit here or expose as flags as needed).
    cfg.width = 1000.0
    cfg.height = 1000.0
    cfg.dt = 1.0
    cfg.coverage = 320.0
    cfg.prefill_speed = 2000.0     # tokens/sec
    cfg.token_rate = 20.0          # tokens/sec of ongoing generation
    cfg.max_context = 4000.0
    cfg.kv_bytes_per_token = 0.2   # MB/token
    cfg.backhaul_bw = 100.0        # MB/s
    cfg.contention_alpha = 0.3
    cfg.pingpong_window = 20.0

    srv_rng = random.Random(cfg.seed + 999)
    servers = build_servers(cfg.servers, cfg.width, cfg.height,
                            cfg.coverage, cfg.prefill_speed, srv_rng)
    trace = precompute_trace(cfg, servers)

    print(f"scenario: {cfg.users} users, {cfg.servers} servers, "
          f"{cfg.steps} steps, speed={cfg.speed}, seed={cfg.seed}\n")
    header = f"{'policy':<20} {'HO':>5} {'mean(s)':>8} {'p99(s)':>7} " \
             f"{'downtime':>9} {'pingpong':>9} {'prefetch':>9}"
    print(header)
    print("-" * len(header))
    for policy in all_policies():
        s = run_policy(policy, cfg, servers, trace)
        print(f"{policy.name:<20} {s['handovers']:>5} {s['mean_delay_s']:>8} "
              f"{s['p99_delay_s']:>7} {s['total_downtime_s']:>9} "
              f"{s['pingpong']:>9} {s['prefetch_acts']:>9}")


if __name__ == "__main__":
    main()
