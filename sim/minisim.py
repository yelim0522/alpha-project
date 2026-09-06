"""Single-target mini simulator: K users on server 0 are predicted to hand over to
server 1 at given instants. Used for the Pallas reproduction (all deadlines equal)
and for the small-instance optimality-gap study (heterogeneous deadlines)."""

from typing import Dict, List, Sequence

from environment import (Server, User, CostParams, LinkLoad, MODEL_PRESETS, advance_preparations,
                         edf_key)
from metrics import Metrics
from prediction import Prediction


def params_for(model: str, activation_serial: float = 0.04) -> CostParams:
    p = MODEL_PRESETS[model]
    return CostParams(kv_mb_per_token=p["kv_kib"] / 1024.0, decode_rate=p["decode_rate"],
                      activation_latency=p["t0_reactive"],
                      activation_latency_proactive=p["t0_proactive"],
                      activation_serial=activation_serial, itl_base_ms=p["itl_ms"])


def mini_sim(policy, model: str, link_mbps: float, contexts: Sequence[float],
             t_hos: Sequence[float], dt: float, vram_mb: float = 20000.0,
             prefill_parallel: float = 4.0, with_early: bool = False):
    """Return per-user SIT (s) (and per-user early exposure if with_early). Users
    whose handover instant coincides share the target GPU/link at that instant;
    prepared caches activate serially."""
    p, params = MODEL_PRESETS[model], params_for(model)
    B = link_mbps / 8.0
    servers: Dict[int, Server] = {
        0: Server(0, 0.0, 0.0, 1.0, p["prefill_speed"], B, vram_mb, prefill_parallel),
        1: Server(1, 1.0, 0.0, 1.0, p["prefill_speed"], B, vram_mb, prefill_parallel),
    }
    users = [User(id=i, x=0.0, y=0.0, tokens=float(L), server=0, anchor=0)
             for i, L in enumerate(contexts)]
    preds = {u.id: Prediction(1, float(t_hos[u.id]), 1.0) for u in users}
    metrics = Metrics()
    policy.reset()
    policy.trigger_log = []
    loads = {sid: LinkLoad() for sid in servers}
    horizon = max(t_hos)
    n_steps = int(round(horizon / dt))
    sits = [None] * len(users)
    earlies = [0.0] * len(users)
    t = 0.0
    for step in range(n_steps + 1):
        # Handovers due at this instant (resolved before the control cycle).
        due = [u for u in users if sits[u.id] is None and preds[u.id].t_ho <= t + 1e-9]
        if due:
            tgt = servers[1]
            n_gpu = sum(1 for u in due if u.prep is None or u.prep.target != 1)
            n_bg = 0 if policy.order_key is edf_key else loads[1].prefills
            v_share = tgt.prefill_share(n_gpu + n_bg)
            bw_share = tgt.backhaul_bw / max(1, len(due) + loads[1].streams)
            activations = 0
            for u in due:
                dec = policy.plan_handover(u, tgt, params, v_share, bw_share, False, preds[u.id], t)
                if dec.used_prep:
                    dec.sit += activations * params.activation_serial
                    activations += 1
                if u.prep is not None and u.prep.target == 1 and u.prep.prefix_done_t is not None:
                    earlies[u.id] = max(0.0, t - u.prep.prefix_done_t)
                sits[u.id] = dec.sit
                u.prep = None
                u.server = u.anchor = 1
                preds.pop(u.id, None)
        if step == n_steps:
            break
        policy.on_step(t, dt, users, servers, params, preds, loads, metrics)
        loads = advance_preparations(users, servers, params, t, dt, policy.order_key)
        for u in users:
            if sits[u.id] is None:
                u.tokens += params.decode_rate * dt
        t += dt
    sits = [s if s is not None else 0.0 for s in sits]
    return (sits, earlies) if with_early else sits
