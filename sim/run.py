"""Entry point: builds a scenario, precomputes one shared mobility/token/prediction
trace, then replays it under every policy for a paired comparison.

Run:  python3 run.py
      python3 run.py --mobility rwp                 # independent users
      python3 run.py --users 128 --group 16         # denser platoons (herding up)
      python3 run.py --sweep-users 32,64,128,192    # density sweep
      python3 run.py --controlled                   # Pallas re-tuning variants
      python3 run.py --ablation                     # coordinated mechanism ladder
      python3 run.py --model qwen14b --seeds 5      # other model preset, seed average
Model presets (c, prefill rate, decode rate, T0) are calibrated to the numbers
published for Pallas; see environment.MODEL_PRESETS and reproduce_pallas.py.
"""

import argparse
import random
from typing import Dict, List

from environment import (
    Server, User, CostParams, LinkLoad, MODEL_PRESETS, nearest_server, advance_preparations,
    edf_key,
)
from mobility import RandomWaypoint, GroupFlow
from metrics import Metrics
from policies import all_policies, ablation_ladder, PallasApprox, Coordinated, Decision
from prediction import predict_all


def make_params(cfg) -> CostParams:
    return CostParams(kv_mb_per_token=cfg.kv_mb_per_token, decode_rate=cfg.decode_rate,
                      activation_latency=cfg.t0_reactive,
                      activation_latency_proactive=cfg.t0_proactive,
                      activation_serial=cfg.activation_serial, itl_base_ms=cfg.itl_ms)


def build_servers(cfg, rng) -> List[Server]:
    m = cfg.servers
    cols = int(m ** 0.5 + 0.999) or 1
    rows = (m + cols - 1) // cols
    servers, idx = [], 0
    for r in range(rows):
        for c in range(cols):
            if idx >= m:
                break
            servers.append(Server(idx, (c + 0.5) * cfg.width / cols, (r + 0.5) * cfg.height / rows,
                                  cfg.coverage, cfg.prefill_speed, cfg.backhaul_bw, cfg.vram_mb,
                                  cfg.prefill_parallel))
            idx += 1
    return servers


def precompute_trace(cfg, servers):
    """Identical per-step positions, token counts and predictions for all policies."""
    rng = random.Random(cfg.seed)
    if cfg.mobility == "flow":
        mob = GroupFlow(cfg.width, cfg.height, cfg.speed, rng, group_size=cfg.group)
    else:
        mob = RandomWaypoint(cfg.width, cfg.height, cfg.speed, rng)
    users = [User(id=i, x=rng.uniform(0, cfg.width), y=rng.uniform(0, cfg.height),
                  tokens=rng.uniform(cfg.min_context, cfg.max_context * 0.7))
             for i in range(cfg.users)]
    pred_rng = random.Random(cfg.seed + 7)
    trace = []
    for t_idx in range(cfg.steps):
        t = t_idx * cfg.dt
        snapshot = []
        for u in users:
            mob.step(u, cfg.dt)
            if u.respawned:
                # New session for a user re-entering the area.
                u.tokens = rng.uniform(cfg.min_context, cfg.max_context * 0.7)
            u.tokens = min(cfg.max_context, u.tokens + cfg.decode_rate * cfg.dt)
            u.server = nearest_server(u, servers).id
            snapshot.append((u.x, u.y, u.vx, u.vy, u.tokens, u.respawned))
            u.respawned = False
        preds = predict_all(users, servers, t, cfg.pred_horizon, cfg.pred_step, pred_rng,
                            cfg.pred_speed_noise, cfg.pred_heading_noise)
        trace.append((snapshot, preds))
    return trace


def run_policy(policy, cfg, server_list, trace) -> dict:
    policy.reset()
    servers: Dict[int, Server] = {s.id: s for s in server_list}
    params = make_params(cfg)
    users = [User(id=i, x=0.0, y=0.0) for i in range(cfg.users)]
    metrics = Metrics()
    prev_loads = {sid: LinkLoad() for sid in servers}
    prev_preds: dict = {}
    c = params.kv_mb_per_token

    for t_idx, (snap, preds) in enumerate(trace):
        t = t_idx * cfg.dt
        pending: Dict[int, list] = {}
        for u, (x, y, vx, vy, tokens, respawned) in zip(users, snap):
            u.x, u.y, u.vx, u.vy, u.tokens = x, y, vx, vy, tokens
            here = nearest_server(u, servers.values())
            if u.server == -1 or respawned:
                # Fresh session: state is born at the serving server, nothing to migrate.
                if u.prep is not None:
                    u.prep, u.epoch = None, u.epoch + 1
                u.server = u.anchor = here.id
                u.hops = 0
                u.history = [(here.id, t)]
            elif here.id != u.server:
                pending.setdefault(here.id, []).append((u, here))

        # 1) Resolve handovers detected at this step. The decision uses the prediction
        #    issued *before* the crossing (prev_preds) and the resource state carried
        #    over from the previous cycle, i.e. what the target actually sees now.
        loads = {sid: LinkLoad() for sid in servers}
        for sid, lst in pending.items():
            srv, ld, bg = servers[sid], loads[sid], prev_loads[sid]
            n_rec = len(lst)
            # GPU is shared by users recovering reactively (no usable preparation)
            # plus the prefills already in flight (which include unfinished prefixes
            # of the users handing over now). Activating a finished prefix is not a
            # prefill and does not dilute the share.
            n_gpu = sum(1 for u, _ in lst if u.prep is None or u.prep.target != sid)
            # Under EDF service the recoveries (deadline = now) pre-empt in-flight
            # preparations; under FCFS they queue behind them.
            n_bg = 0 if policy.order_key is edf_key else bg.prefills
            v_share = srv.prefill_share(n_gpu + n_bg)
            bw_share = srv.backhaul_bw / max(1, n_rec + bg.streams)
            activations = 0
            for u, dst in lst:
                is_pingpong = any(s == dst.id and t - ts <= cfg.pingpong_window
                                  for s, ts in u.history[:-1])
                if dst.id == u.anchor:
                    dec = Decision(0.0, 0.0, 0.0, "return", False)
                else:
                    dec = policy.plan_handover(u, dst, params, v_share, bw_share,
                                               is_pingpong, prev_preds.get(u.id), t)
                if dec.used_prep:
                    # Prepared requests are activated one after another at the target.
                    dec.sit += activations * params.activation_serial
                    activations += 1
                early = None
                if u.prep is not None and u.prep.target == dst.id and u.prep.prefix_done_t is not None:
                    early = t - u.prep.prefix_done_t
                metrics.record_handover(dec, is_pingpong, early, u.id)
                ld.demand_mb += dec.transfer_mb
                ld.sent_mb += dec.transfer_mb
                if dec.transfer_mb > 0:
                    ld.streams += 1
                if dec.recompute_tokens > 0:
                    ld.prefills += 1
                if dec.mode == "migrate":
                    u.anchor, u.hops = dst.id, 0
                elif dec.mode == "detour":
                    u.hops += 1
                else:
                    u.hops = 0
                if u.prep is not None:
                    if dec.used_prep:
                        u.prep = None
                    else:
                        policy._cancel(u, params, metrics, t)
                u.server = dst.id
                u.history.append((dst.id, t))
                if len(u.history) > 16:
                    u.history.pop(0)

        # 2) Policy control cycle with the fresh predictions, then resource progression.
        policy.on_step(t, cfg.dt, users, servers, params, preds, loads, metrics)
        prep_loads = advance_preparations(users, servers, params, t, cfg.dt, policy.order_key)
        for sid, ld in prep_loads.items():
            tot = loads[sid]
            tot.streams += ld.streams
            tot.prefills += ld.prefills
            tot.demand_mb += ld.demand_mb
            tot.sent_mb += ld.sent_mb

        # 3) Background "settle" migrations (state follows a detoured user) complete
        #    once the prefix is rebuilt; the short deferred suffix is prefilled then.
        for u in users:
            p = u.prep
            if p is not None and p.target == u.server and p.prefix_remaining <= 0:
                v_share = servers[u.server].prefill_share(loads[u.server].prefills + 1)
                sit = params.activation_latency_proactive + p.unsent_suffix_tokens(c) / v_share
                metrics.record_settle(sit, u.id)
                loads[u.server].prefills += 1
                u.anchor, u.hops, u.prep = u.server, 0, None

        metrics.record_itl(users, params)
        metrics.record_step_load(loads, cfg.dt)
        prev_loads, prev_preds = loads, preds
    return metrics.summary()


# (title, width, summary key, decimals; None decimals = integer)
COLUMNS = [
    ("HO", 5, "handovers", None),
    ("SITavg", 7, "sit_mean_s", 3),
    ("SITp99", 7, "sit_p99_s", 3),
    ("SITmax", 7, "sit_max_s", 3),
    ("prep%", 5, "prep_rate_pct", None),
    ("SITprep", 7, "sit_prep_s", 3),
    ("prp99", 6, "sit_prep_p99_s", 2),
    ("SITnoprp", 8, "sit_noprep_s", 3),
    ("ITLms", 6, "itl_ms", 1),
    ("pkStrm", 6, "peak_streams", None),
    ("CoV", 5, "load_cov", 2),
    ("Jain", 5, "jain", 2),
    ("early", 6, "early_s", 2),
    ("wasteMB", 8, "wasted_mb", 0),
    ("detour", 6, "detours", None),
]


def average(summaries: List[dict]) -> dict:
    return {k: sum(s[k] for s in summaries) / len(summaries) for k in summaries[0]}


def print_table(rows):
    header = f"{'policy':<16} " + " ".join(f"{t:>{w}}" for t, w, _, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for name, s in rows:
        cells = []
        for _, w, key, dec in COLUMNS:
            v = s[key]
            cells.append(f"{int(round(v)):>{w}}" if dec is None else f"{v:>{w}.{dec}f}")
        print(f"{name:<16} " + " ".join(cells))


def policies_for(cfg):
    pols = all_policies()
    if cfg.controlled:
        pols += [PallasApprox(t_max=15.0, label="pallas-tmax15"),
                 PallasApprox(alpha=0.5, label="pallas-a0.5"),
                 PallasApprox(ewma=0.95, t_max=15.0, label="pallas-fastobs"),
                 PallasApprox(edf=True, label="pallas-edf"),
                 Coordinated(social=False, label="coord-nosocial"),
                 Coordinated(beta=0.5, label="coord-b0.5"),
                 Coordinated(beta=2.0, label="coord-b2"),
                 Coordinated(beta=4.0, label="coord-b4"),
                 Coordinated(t_max=15.0, label="coord-tmax15"),
                 Coordinated(stream_util=9.0, label="coord-stream")]
    if cfg.ablation:
        pols += ablation_ladder()
    return pols


def run_scenario(cfg):
    pols = policies_for(cfg)
    per_policy = {p.name: [] for p in pols}
    for k in range(cfg.seeds):
        cfg.seed = cfg.base_seed + k
        servers = build_servers(cfg, random.Random(cfg.seed + 999))
        trace = precompute_trace(cfg, servers)
        for p in pols:
            per_policy[p.name].append(run_policy(p, cfg, servers, trace))
    return [(name, average(v)) for name, v in per_policy.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=64)
    ap.add_argument("--servers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--seed", dest="base_seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--speed", type=float, default=25.0)
    ap.add_argument("--mobility", choices=["flow", "rwp"], default="flow")
    ap.add_argument("--group", type=int, default=8, help="platoon size for --mobility flow")
    ap.add_argument("--backhaul-mbps", type=float, default=300.0)
    ap.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen32b",
                    help="model preset (KV size, prefill/decode rate, T0); individual flags override")
    ap.add_argument("--prefill-speed", type=float, default=None, help="per-request prefill tok/s")
    ap.add_argument("--prefill-parallel", type=float, default=4.0,
                    help="aggregate GPU prefill capacity as a multiple of --prefill-speed")
    ap.add_argument("--decode-rate", type=float, default=None)
    ap.add_argument("--kv-kib", type=float, default=None, help="KV bytes per token (KiB)")
    ap.add_argument("--activation-serial", type=float, default=0.04)
    ap.add_argument("--vram-mb", type=float, default=6000.0)
    ap.add_argument("--pred-speed-noise", type=float, default=0.0)
    ap.add_argument("--pred-heading-noise", type=float, default=0.0)
    ap.add_argument("--controlled", action="store_true", help="add Pallas/coordinated re-tuning variants")
    ap.add_argument("--ablation", action="store_true", help="add the coordinated mechanism ladder")
    ap.add_argument("--sweep-users", type=str, default="", help="comma list, e.g. 32,64,128")
    cfg = ap.parse_args()

    preset = MODEL_PRESETS[cfg.model]
    cfg.kv_kib = cfg.kv_kib if cfg.kv_kib is not None else preset["kv_kib"]
    cfg.prefill_speed = cfg.prefill_speed if cfg.prefill_speed is not None else preset["prefill_speed"]
    cfg.decode_rate = cfg.decode_rate if cfg.decode_rate is not None else preset["decode_rate"]
    cfg.itl_ms = preset["itl_ms"]
    cfg.t0_reactive = preset["t0_reactive"]
    cfg.t0_proactive = preset["t0_proactive"]

    cfg.width = cfg.height = 1000.0
    cfg.coverage = 320.0
    cfg.min_context = 500.0
    cfg.max_context = 4500.0
    cfg.kv_mb_per_token = cfg.kv_kib / 1024.0
    cfg.backhaul_bw = cfg.backhaul_mbps / 8.0     # MB/s
    cfg.pingpong_window = 20.0
    cfg.pred_horizon = 20.0
    cfg.pred_step = cfg.dt

    sweep = [int(x) for x in cfg.sweep_users.split(",") if x] or [cfg.users]
    for n in sweep:
        cfg.users = n
        print(f"scenario: {cfg.users} users ({cfg.mobility}, group={cfg.group}), {cfg.servers} servers, "
              f"{cfg.steps}x{cfg.dt}s, speed={cfg.speed} m/s, link={cfg.backhaul_mbps} Mbps, "
              f"model={cfg.model} (c={cfg.kv_kib:.0f} KiB, v1={cfg.prefill_speed:.0f} tok/s, "
              f"x{cfg.prefill_parallel:.0f} batched), seeds={cfg.seeds}\n")
        print_table(run_scenario(cfg))
        print()


if __name__ == "__main__":
    main()
