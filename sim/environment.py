"""Core entities and cost model for the edge-LLM KV-cache handover simulator.

The physical picture: users move across a 2D plane covered by edge servers. Each
user holds an ongoing LLM conversation whose KV cache grows with the token count.
When the serving server changes (handover), the target server must restore the KV
cache either by transferring it over the backhaul or by recomputing it via prefill.
"""

from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class Server:
    id: int
    x: float
    y: float
    coverage: float          # coverage radius (same distance units as positions)
    prefill_speed: float     # v_m, tokens/sec the server can recompute during prefill


@dataclass
class User:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    tokens: float = 0.0          # L_k(t): conversation length in tokens
    server: int = -1             # current serving server id (-1 = unassigned)
    prefetched: float = 0.0      # tokens already streamed toward prefetch_target
    prefetch_target: int = -1    # server id the prefetch is aimed at
    history: List = field(default_factory=list)  # [(server_id, time), ...] recent handovers


@dataclass
class CostParams:
    kv_bytes_per_token: float = 0.2   # c: KV cache size per token (MB/token)
    backhaul_bw: float = 100.0        # B: nominal backhaul bandwidth (MB/s)
    contention_alpha: float = 0.3     # bandwidth sharing sensitivity to concurrency


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def nearest_server(user: User, servers: List[Server]) -> Server:
    """Nearest server within coverage; fall back to globally nearest if none covers."""
    best, best_d = None, float("inf")
    for s in servers:
        d = distance(user.x, user.y, s.x, s.y)
        if d <= s.coverage and d < best_d:
            best, best_d = s, d
    if best is not None:
        return best
    for s in servers:
        d = distance(user.x, user.y, s.x, s.y)
        if d < best_d:
            best, best_d = s, d
    return best


def effective_bandwidth(nominal_bw: float, concurrency: int, alpha: float) -> float:
    """Backhaul contention: many simultaneous transfers share the link."""
    return nominal_bw / (1.0 + alpha * max(0, concurrency - 1))


def optimal_prefill_length(remaining_tokens: float, c: float, v: float, bw: float) -> float:
    """p* that balances recompute time (p/v) and transfer time (c*(L-p)/bw)."""
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
