"""Core entities, cost model and shared-resource accounting for the edge-LLM
KV-cache handover simulator.

Physical picture: users move across a 2D plane covered by GPU-equipped edge
servers (AI-RAN gNBs). Each user holds an ongoing LLM session whose KV cache
grows with the token count. When the serving server changes (handover), the
target must obtain the KV cache by recomputing it (prefill) and/or receiving it
over the inter-gNB backhaul. Proactive policies may start this preparation
*before* the handover toward a predicted target.

Resource model (this is where multi-user effects live):
  - Each server m has one ingress backhaul link of capacity B_m (MB/s) and one
    GPU. A single prefill achieves v_m tokens/s (per-request rate, limited by
    kernel-launch/underutilisation at a few thousand tokens); concurrent
    prefills are batched, so the GPU sustains up to `prefill_parallel` x v_m in
    aggregate before per-request rates start to shrink. This two-level model is
    what Pallas's own concurrency data implies (ctHO worst-user grows only
    442 -> 507 ms from K=1 to K=4 at 1 Gbps, i.e. prefill is not yet shared).
  - Prefill is served in an order chosen by the policy (FCFS by default, EDF for
    the coordinated policy); suffix streams share the link fairly (TCP-like).
  - Activating a prepared request (final block sync + assembly + first decode
    step) costs T0_proactive and is serialised per target: the i-th request
    activated in the same control cycle waits an extra (i-1) x activation_serial.
  - A preparation (`Prep`) follows the Pallas structure: at trigger time the
    context is split into a stable prefix (recomputed at the target) and an
    evolving suffix (tokens generated after the trigger), which is either
    streamed as KV blocks or left to be recomputed/transferred at handover.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import math


@dataclass
class Server:
    id: int
    x: float
    y: float
    coverage: float          # coverage radius (m)
    prefill_speed: float     # v_m, tokens/s a single prefill request achieves
    backhaul_bw: float       # B_m, MB/s ingress capacity of the inter-gNB link
    vram_budget_mb: float    # VRAM the server lends to early-prepared migration state
    prefill_parallel: float = 4.0   # aggregate GPU prefill capacity = prefill_parallel * v_m

    def prefill_share(self, n: int) -> float:
        """Per-request prefill rate when n requests prefill concurrently."""
        if n <= 0:
            return self.prefill_speed
        return min(self.prefill_speed, self.prefill_speed * self.prefill_parallel / n)


@dataclass
class Prep:
    """Preparation state of one user toward one predicted target."""
    target: int
    epoch: int
    trigger_t: float
    deadline_t: float               # predicted handover time (may be revised)
    prefix_total: float             # tokens in the stable prefix
    prefix_remaining: float         # prefix tokens still to prefill at the target
    stream_suffix: bool = True      # stream suffix KV blocks (True) or defer them
    suffix_tokens: float = 0.0      # tokens generated since the trigger
    suffix_sent_mb: float = 0.0
    suffix_backlog_mb: float = 0.0  # generated-but-unsent suffix KV (stream mode)
    prefix_done_t: Optional[float] = None

    def vram_mb(self, c: float) -> float:
        return (self.prefix_total - self.prefix_remaining) * c + self.suffix_sent_mb

    def unsent_suffix_tokens(self, c: float) -> float:
        if self.stream_suffix:
            return self.suffix_backlog_mb / c if c > 0 else 0.0
        return self.suffix_tokens


@dataclass
class User:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    tokens: float = 0.0          # L_k(t): context length in tokens
    server: int = -1             # current serving (radio) server
    anchor: int = -1             # server that holds the inference state
    hops: int = 0                # forwarding hops between anchor and serving server
    epoch: int = 0
    prep: Optional[Prep] = None
    respawned: bool = False      # set by the mobility model when the user re-enters the area
    history: List = field(default_factory=list)  # [(server_id, time), ...]


@dataclass
class CostParams:
    kv_mb_per_token: float = 0.25          # c: 256 KiB/token (Qwen3-32B, BF16)
    decode_rate: float = 15.0              # r_d: tokens/s generated per session
    activation_latency: float = 0.075      # T0 of post-handover recovery paths (s)
    activation_latency_proactive: float = 0.32  # T0 of a prepared request: final sync + assembly
    activation_serial: float = 0.04        # extra wait per additional activation in the same cycle
    itl_base_ms: float = 67.0              # ITL when served locally
    itl_hop_penalty_ms: float = 40.0       # extra ITL per forwarding hop (Detour)


# Per-model presets. c follows the architecture (2*layers*kv_heads*head_dim*2B);
# v1 and T0 are fitted to Pallas's published single-user numbers (Table 1 for
# Qwen3-32B; ctHO K=1 and Pallas K=1 in Fig. 8a for Qwen3-14B); Llama-3-8B is
# extrapolated from Table 2 ratios. decode_rate/itl follow Table 2 ITL.
MODEL_PRESETS = {
    "qwen32b": dict(kv_kib=256.0, prefill_speed=1970.0, decode_rate=15.0, itl_ms=67.0,
                    t0_reactive=0.075, t0_proactive=0.32),
    "qwen14b": dict(kv_kib=160.0, prefill_speed=4700.0, decode_rate=22.0, itl_ms=45.5,
                    t0_reactive=0.075, t0_proactive=0.236),
    "llama8b": dict(kv_kib=128.0, prefill_speed=7600.0, decode_rate=40.0, itl_ms=25.0,
                    t0_reactive=0.06, t0_proactive=0.154),
}


@dataclass
class LinkLoad:
    """Per-step resource accounting for one server."""
    streams: int = 0            # concurrent suffix streams (+ recovery transfers)
    prefills: int = 0           # concurrent prefix prefills (+ recovery prefills)
    demand_mb: float = 0.0      # MB wanting to cross the ingress link this step
    sent_mb: float = 0.0        # MB actually carried this step


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def nearest_server_at(x: float, y: float, servers: List[Server]) -> Server:
    """Nearest server within coverage; fall back to globally nearest if none covers."""
    best, best_d = None, float("inf")
    for s in servers:
        d = distance(x, y, s.x, s.y)
        if d <= s.coverage and d < best_d:
            best, best_d = s, d
    if best is not None:
        return best
    for s in servers:
        d = distance(x, y, s.x, s.y)
        if d < best_d:
            best, best_d = s, d
    return best


def nearest_server(user: User, servers: List[Server]) -> Server:
    return nearest_server_at(user.x, user.y, servers)


# --------------------------------------------------------------------------- #
# Single-handover cost model (closed form, used by reactive policies and as the
# residual step of proactive policies)
# --------------------------------------------------------------------------- #

def optimal_prefill_length(remaining_tokens: float, c: float, v: float, bw: float) -> float:
    """p* balancing recompute time p/v against transfer time c*(L-p)/bw (ctHO)."""
    denom = bw + c * v
    if denom <= 0:
        return 0.0
    p = c * v * remaining_tokens / denom
    return max(0.0, min(remaining_tokens, p))


def handover_delay(recompute_tokens: float, transfer_tokens: float,
                   c: float, v: float, bw: float) -> float:
    """Recompute and transfer run in parallel, so delay is the max of the two."""
    recompute_time = recompute_tokens / v if v > 0 else float("inf")
    transfer_time = c * transfer_tokens / bw if bw > 0 else float("inf")
    return max(recompute_time, transfer_time)


def residual_recovery(prefix_tokens: float, free_tokens: float, c: float, v: float,
                      bw: float, allow_split: bool):
    """Post-handover work left after a preparation.

    prefix_tokens : tokens that *must* be prefilled at the target
    free_tokens   : unsent suffix tokens; transferred as KV (allow_split=False, as
                    in Pallas) or split between prefill and transfer (allow_split=True)
    Returns (delay_s, recompute_tokens, transfer_tokens).
    """
    if not allow_split:
        d = handover_delay(prefix_tokens, free_tokens, c, v, bw)
        return d, prefix_tokens, free_tokens
    denom = bw + c * v
    p = (c * v * free_tokens - bw * prefix_tokens) / denom if denom > 0 else 0.0
    p = max(0.0, min(free_tokens, p))
    d = handover_delay(prefix_tokens + p, free_tokens - p, c, v, bw)
    return d, prefix_tokens + p, free_tokens - p


# --------------------------------------------------------------------------- #
# Shared-resource progression of all active preparations for one timestep
# --------------------------------------------------------------------------- #

OrderKey = Callable[[Prep, float], float]


def fcfs_key(prep: Prep, t: float) -> float:
    return prep.trigger_t


def edf_key(prep: Prep, t: float) -> float:
    return prep.deadline_t


def advance_preparations(users: List[User], servers: Dict[int, Server], params: CostParams,
                         t: float, dt: float, order_key: OrderKey = fcfs_key
                         ) -> Dict[int, LinkLoad]:
    """Advance every active Prep by dt under shared GPU/link capacity."""
    loads: Dict[int, LinkLoad] = {sid: LinkLoad() for sid in servers}
    by_target: Dict[int, List[Prep]] = {}
    for u in users:
        if u.prep is not None:
            by_target.setdefault(u.prep.target, []).append(u.prep)

    c = params.kv_mb_per_token
    for sid, preps in by_target.items():
        srv = servers[sid]
        load = loads[sid]

        # GPU: each request is capped at v*dt tokens per step; the batch as a whole
        # at prefill_parallel*v*dt, handed out in policy-defined order.
        cap = srv.prefill_speed * dt
        budget = cap * srv.prefill_parallel
        queue = sorted((p for p in preps if p.prefix_remaining > 0),
                       key=lambda p: order_key(p, t))
        load.prefills = len(queue)
        for p in queue:
            if budget <= 0:
                break
            done = min(p.prefix_remaining, cap, budget)
            p.prefix_remaining -= done
            budget -= done
            if p.prefix_remaining <= 1e-9:
                p.prefix_remaining = 0.0
                p.prefix_done_t = t + dt

        # Source keeps decoding: suffix grows for every prep.
        for p in preps:
            gen = params.decode_rate * dt
            p.suffix_tokens += gen
            if p.stream_suffix:
                p.suffix_backlog_mb += gen * c

        # Link: fair share among streams with backlog.
        streams = [p for p in preps if p.stream_suffix and p.suffix_backlog_mb > 0]
        load.streams = len(streams)
        if streams:
            share = srv.backhaul_bw * dt / len(streams)
            for p in streams:
                load.demand_mb += p.suffix_backlog_mb
                sent = min(p.suffix_backlog_mb, share)
                p.suffix_backlog_mb -= sent
                p.suffix_sent_mb += sent
                load.sent_mb += sent
    return loads


def vram_in_use(users: List[User], target: int, c: float) -> float:
    return sum(u.prep.vram_mb(c) for u in users
               if u.prep is not None and u.prep.target == target)
