"""Policy-dependent response/think clocks for the turn-boundary audit.

Mobility is shared, while each policy consumes the same per-user ordered workload
at its own pace. Empirical output lengths are marginals, not reconstructed chats.
Think times are explicitly synthetic unless supplied as paired CSV observations.
"""

import bisect
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass

from metrics import percentile


class Workload:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lengths, self.weights, self.pairs = [], [], []
        if cfg.turn_workload:
            with open(cfg.turn_workload) as fh:
                if cfg.turn_workload.endswith(".json"):
                    data = json.load(fh)
                    total = 0
                    for length, count in data["output_token_histogram"]:
                        if length <= 0 or count <= 0:
                            raise ValueError("workload lengths/counts must be positive")
                        total += count
                        self.lengths.append(float(length))
                        self.weights.append(total)
                else:
                    for row in csv.DictReader(fh):
                        length, gap = float(row["output_tokens"]), float(row["think_s"])
                        if not math.isfinite(length + gap) or length <= 0 or gap < 0:
                            raise ValueError("invalid output_tokens/think_s")
                        self.pairs.append((length, gap))
            if not self.lengths and not self.pairs:
                raise ValueError("empty turn workload")

    def sample(self, output_rng, think_rng):
        c = self.cfg
        if self.pairs:
            return output_rng.choice(self.pairs)
        if self.lengths:
            length = self.lengths[bisect.bisect_right(
                self.weights, output_rng.randrange(self.weights[-1]))]
        elif c.turn_distribution == "lognormal":
            length = max(1.0, output_rng.lognormvariate(
                math.log(c.turn_output_tokens) - 0.5, 1.0))
        else:
            length = c.turn_output_tokens
        if c.think_distribution == "fixed" or c.think_time == 0:
            gap = c.think_time
        else:
            # Mixture has 25% immediate follow-ups; both variants have the
            # requested *population mean*, without silently clipping long tails.
            p_zero = 0.25 if c.think_distribution == "mixture" else 0.0
            gap = (0.0 if think_rng.random() < p_zero else think_rng.lognormvariate(
                math.log(c.think_time / (1 - p_zero)) - 0.5, 1.0))
        return length, gap


@dataclass
class Conversation:
    output_rng: object
    think_rng: object
    remaining_tokens: float
    gap: float
    phase: str = "generate"
    think_left: float = 0.0
    request_t: object = None  # None marks the left-censored initial response
    response_tokens: float = 0.0
    response_wait: float = 0.0
    stall: float = 0.0


class ConversationClock:
    def __init__(self, cfg, metrics):
        self.cfg, self.metrics = cfg, metrics
        self.workload = Workload(cfg)
        self.states = {}
        self.sessions = Counter()
        self.totals = Counter()
        self.waits, self.response_waits, self.response_excess = [], [], []
        self.response_e2e = []

    def reset_user(self, u):
        if u.id in self.states:
            self._censor(self.states[u.id])
        session = self.sessions[u.id]
        self.sessions[u.id] += 1
        seed = self.cfg.seed * 1000000007 + u.id * 100003 + session * 1009
        out_rng, gap_rng = random.Random(seed + 41), random.Random(seed + 53)
        length, gap = self.workload.sample(out_rng, gap_rng)
        s = Conversation(out_rng, gap_rng, length, gap)
        # Same initial phase for every policy, with a separate RNG from workloads.
        phase = random.Random(seed + 67).uniform(0, length / self.cfg.decode_rate + gap)
        if phase < length / self.cfg.decode_rate:
            s.remaining_tokens = length - phase * self.cfg.decode_rate
        else:
            s.phase, s.remaining_tokens = "think", 0.0
            s.think_left = length / self.cfg.decode_rate + gap - phase
        self.states[u.id] = s
        u.recovery_until_s = 0.0
        u.boundary_ready_t = 0.0
        self.sync(u)

    def sync(self, u):
        s = self.states[u.id]
        u.turn_remaining_s = s.remaining_tokens / self.cfg.decode_rate if s.phase == "generate" else 0.0
        u.think_remaining_s = s.think_left if s.phase == "think" else 0.0
        u.next_think_s = s.gap
        u.boundary_crossed_step = False
        u.boundary_offset_s = u.boundary_gap_s = 0.0
        u.generated_tokens_step = 0.0

    def _finish_stall(self, s, censored=False):
        if s.stall > 1e-9:
            if censored:
                self.totals["censored_wait_s"] += s.stall
                self.totals["censored_wait_episodes"] += 1
            else:
                self.waits.append(s.stall)
            s.stall = 0.0

    def _censor(self, s):
        self._finish_stall(s, censored=True)
        if s.request_t is not None:
            self.totals["censored_responses"] += 1

    def _wait(self, s, duration):
        s.stall += duration
        s.response_wait += duration
        self.totals["observed_wait_s"] += duration

    def account_residence(self, users, dt, params):
        resident = Counter()
        for u in users:
            resident[u.anchor] += u.tokens * params.kv_mb_per_token
            if u.anchor != u.server:
                self.totals["retained_kv_gb_s"] += u.tokens * params.kv_mb_per_token * dt / 1000
                self.totals["remote_session_s"] += dt
        self.totals["peak_resident_kv_mb"] = max(
            self.totals["peak_resident_kv_mb"], max(resident.values(), default=0.0))

    def advance(self, users, policy, t, dt, params):
        boundary = policy.name == "turn-boundary"
        ready = {}
        for u in users:
            # In-flight jobs may complete inside this step. Their exact finish
            # time gates the next response; a detached generation may finish first.
            ready[u.id] = (float("inf") if boundary and u.anchor != u.server
                           else u.boundary_ready_t)
        candidates = Counter()
        for u in users:
            s = self.states[u.id]
            eligible = s.phase == "generate" or ready[u.id] < t + dt
            if eligible and u.recovery_until_s < t + dt and (s.phase != "think" or s.think_left < dt):
                candidates[u.anchor] += 1
        for u in users:
            period = 1.0 / params.decode_rate + params.itl_hop_penalty_ms * u.hops / 1000
            rate = 1.0 / period
            if self.cfg.decode_capacity > 0:
                rate = min(rate, self.cfg.decode_capacity / max(1, candidates[u.anchor]))
            self._advance_user(u, t, dt, rate, ready[u.id], boundary)

    def _advance_user(self, u, t, dt, rate, boundary_ready, boundary):
        s, end, now = self.states[u.id], t + dt, t
        generated = 0.0
        while now < end - 1e-9:
            if s.phase == "think":
                span = min(end - now, s.think_left)
                s.think_left -= span
                now += span
                if s.think_left <= 1e-9:
                    length, gap = self.workload.sample(s.output_rng, s.think_rng)
                    s.remaining_tokens, s.gap, s.response_tokens = length, gap, length
                    s.request_t, s.response_wait, s.phase = now, 0.0, "ready"
                continue
            gate = u.recovery_until_s
            if boundary and s.phase == "ready":
                gate = max(gate, boundary_ready)
            if gate > now + 1e-9:
                span = min(end, gate) - now
                self._wait(s, span)
                now += span
                continue
            self._finish_stall(s)
            s.phase = "generate"
            span = min(end - now, s.remaining_tokens / rate)
            tokens = min(s.remaining_tokens, span * rate)
            s.remaining_tokens -= tokens
            generated += tokens
            now += span
            self.metrics.itl_sum_ms += span * 1000
            self.metrics.itl_samples += tokens
            if u.anchor != u.server:
                self.totals["forwarded_tokens"] += tokens
            if s.remaining_tokens <= 1e-9:
                if s.request_t is not None:
                    e2e = now - s.request_t
                    self.response_e2e.append(e2e)
                    self.response_excess.append(max(0.0, e2e - s.response_tokens / self.cfg.decode_rate))
                    self.response_waits.append(s.response_wait)
                s.phase, s.think_left, s.request_t = "think", s.gap, None
        u.tokens = min(self.cfg.max_context, u.tokens + generated)
        u.generated_tokens_step = generated
        self.totals["generated_tokens"] += generated

    def summary(self):
        result = {k: self.totals[k] for k in (
            "observed_wait_s", "censored_wait_s", "censored_wait_episodes",
            "censored_responses", "generated_tokens", "forwarded_tokens",
            "retained_kv_gb_s", "remote_session_s", "peak_resident_kv_mb")}
        for s in self.states.values():
            result["censored_wait_s"] += s.stall
            result["censored_wait_episodes"] += int(s.stall > 1e-9)
            result["censored_responses"] += int(s.request_t is not None)
        result["completed_wait_s"] = sum(self.waits)
        result["responses_completed"] = len(self.response_e2e)
        for name, values in (("wait_episode", self.waits), ("response_wait", self.response_waits),
                             ("response_excess", self.response_excess), ("response_e2e", self.response_e2e)):
            result[name + "_mean_s"] = sum(values) / len(values) if values else 0.0
            result[name + "_p99_s"] = percentile(values, 99)
        return result
