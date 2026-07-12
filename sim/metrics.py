"""Metric collection, structured by role (mirrors the paper's layered validation).

Primary  : handover delay (mean/p99), ping-pong count  -> structural symptom.
Secondary: prefetch activation count                    -> causal link of the fix.
Auxiliary: total downtime, transferred volume, handover count.
"""

from dataclasses import dataclass, field
from typing import List


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1))))
    return s[idx]


@dataclass
class Metrics:
    handover_delays: List[float] = field(default_factory=list)
    pingpong_count: int = 0
    prefetch_activations: int = 0
    handover_count: int = 0
    transferred_tokens: float = 0.0

    def record_handover(self, delay, transfer_tokens, is_pingpong, used_prefetch):
        self.handover_delays.append(delay)
        self.handover_count += 1
        self.transferred_tokens += transfer_tokens
        if is_pingpong:
            self.pingpong_count += 1
        if used_prefetch:
            self.prefetch_activations += 1

    def summary(self) -> dict:
        delays = self.handover_delays
        mean = sum(delays) / len(delays) if delays else 0.0
        return {
            "handovers": self.handover_count,
            "mean_delay_s": round(mean, 3),
            "p99_delay_s": round(percentile(delays, 99), 3),
            "total_downtime_s": round(sum(delays), 1),
            "pingpong": self.pingpong_count,
            "prefetch_acts": self.prefetch_activations,
            "transfer_Mtok": round(self.transferred_tokens / 1e6, 3),
        }
