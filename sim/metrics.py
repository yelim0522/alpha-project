"""Metric collection, layered as in the paper's validation design.

Primary (user-facing, comparable with Pallas/ctHO):
    SIT mean / p99 / worst-user, ITL mean (Detour penalty), total downtime.
Structural (multi-user, the mechanism the proposal targets):
    peak concurrent streams on any link, peak link demand (MB/s),
    coefficient of variation of time-axis link demand (burstiness),
    early-preparation exposure T_early, wasted (cancelled) preparation.
Causal / bookkeeping:
    handovers, ping-pong count, preparations actually used, detours, settles,
    total transferred MB.
"""

from dataclasses import dataclass, field
from typing import Dict, List


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return s[idx]


@dataclass
class Metrics:
    sits: List[float] = field(default_factory=list)
    early_s: List[float] = field(default_factory=list)
    handover_count: int = 0
    migrations: int = 0
    detours: int = 0
    returns: int = 0
    settles: int = 0
    pingpong_count: int = 0
    prep_used: int = 0
    cancels: int = 0
    wasted_mb: float = 0.0
    transferred_mb: float = 0.0
    itl_sum_ms: float = 0.0
    itl_samples: int = 0
    step_demand_mb: List[float] = field(default_factory=list)
    peak_streams: int = 0
    peak_link_mb: float = 0.0
    per_user_sit: Dict[int, List[float]] = field(default_factory=dict)
    sits_prepared: List[float] = field(default_factory=list)    # migrations that used a preparation
    sits_unprepared: List[float] = field(default_factory=list)  # reactive fallbacks

    def record_handover(self, decision, is_pingpong: bool, early, user_id: int = -1):
        self.handover_count += 1
        self.transferred_mb += decision.transfer_mb
        if decision.mode == "migrate":
            self.migrations += 1
            self.sits.append(decision.sit)
            self.per_user_sit.setdefault(user_id, []).append(decision.sit)
            (self.sits_prepared if decision.used_prep else self.sits_unprepared).append(decision.sit)
        elif decision.mode == "detour":
            self.detours += 1
        else:
            self.returns += 1
        if is_pingpong:
            self.pingpong_count += 1
        if decision.used_prep:
            self.prep_used += 1
        if early is not None:
            self.early_s.append(max(0.0, early))

    def record_settle(self, sit: float, user_id: int = -1):
        self.settles += 1
        self.sits.append(sit)
        self.per_user_sit.setdefault(user_id, []).append(sit)

    def jain_index(self) -> float:
        """Jain fairness over per-user mean SIT (1 = perfectly even)."""
        means = [sum(v) / len(v) for v in self.per_user_sit.values() if v]
        if not means:
            return 1.0
        s1 = sum(means)
        s2 = sum(m * m for m in means)
        return (s1 * s1) / (len(means) * s2) if s2 > 0 else 1.0

    def record_cancel(self, prep, c: float):
        self.cancels += 1
        self.wasted_mb += prep.vram_mb(c)

    def record_itl(self, users, params):
        for u in users:
            self.itl_sum_ms += params.itl_base_ms + params.itl_hop_penalty_ms * u.hops
            self.itl_samples += 1

    def record_step_load(self, loads: Dict[int, object], dt: float):
        total = 0.0
        for ld in loads.values():
            total += ld.demand_mb
            self.peak_streams = max(self.peak_streams, ld.streams)
            self.peak_link_mb = max(self.peak_link_mb, ld.demand_mb / dt)
        self.step_demand_mb.append(total / dt)

    def summary(self) -> dict:
        sits = self.sits
        mean = sum(sits) / len(sits) if sits else 0.0
        n = len(self.step_demand_mb)
        if n > 1:
            mu = sum(self.step_demand_mb) / n
            var = sum((x - mu) ** 2 for x in self.step_demand_mb) / (n - 1)
            cov = (var ** 0.5) / mu if mu > 0 else 0.0
        else:
            cov = 0.0
        def avg(v):
            return sum(v) / len(v) if v else 0.0
        return {
            "handovers": self.handover_count,
            "sit_mean_s": mean,
            "prep_rate_pct": 100.0 * len(self.sits_prepared) / self.migrations if self.migrations else 0.0,
            "sit_prep_s": avg(self.sits_prepared),
            "sit_prep_p99_s": percentile(self.sits_prepared, 99),
            "sit_noprep_s": avg(self.sits_unprepared),
            "sit_p99_s": percentile(sits, 99),
            "sit_max_s": max(sits) if sits else 0.0,
            "downtime_s": sum(sits),
            "itl_ms": self.itl_sum_ms / self.itl_samples if self.itl_samples else 0.0,
            "pingpong": self.pingpong_count,
            "detours": self.detours,
            "prep_used": self.prep_used,
            "cancels": self.cancels,
            "wasted_mb": self.wasted_mb,
            "early_s": sum(self.early_s) / len(self.early_s) if self.early_s else 0.0,
            "peak_streams": self.peak_streams,
            "peak_mbps": self.peak_link_mb,
            "load_cov": cov,
            "jain": self.jain_index(),
            "transfer_mb": self.transferred_mb,
        }
