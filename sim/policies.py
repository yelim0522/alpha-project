"""Handover state-transfer policies.

Each policy decides, at a handover, how many tokens to recompute (prefill) vs.
transfer over the backhaul. The proposed policy additionally prefetches KV cache
before the handover based on a mobility prediction, and gates its intervention by
recent mobility/contention so it stays idle in low-load regimes.

Baselines:
  - AlwaysTransfer   : transfer the whole remaining KV cache (p = 0).
  - AlwaysRecompute  : recompute the whole remaining KV cache (p = L).
  - ReactiveHybrid   : pick the optimal balance p* at handover time (target-paper style).
Proposed:
  - PredictivePrefetch: ReactiveHybrid + proactive prefetch + ping-pong gating.
"""

import math
from typing import List

from environment import (
    Server, User, CostParams, nearest_server, distance,
    optimal_prefill_length, handover_delay,
)


class Policy:
    name = "base"

    def on_step(self, t, dt, users, servers, params, gate_level):
        """Called every timestep. Baselines do nothing; prefetch policies act here."""
        return 0  # number of prefetch activations this step

    def plan_handover(self, user, dst, params, bw_eff):
        """Return (delay, recompute_tokens, transfer_tokens, used_prefetch)."""
        raise NotImplementedError

    def reset(self):
        pass


class AlwaysTransfer(Policy):
    name = "always-transfer"

    def plan_handover(self, user, dst, params, bw_eff):
        remaining = user.tokens
        d = handover_delay(0.0, remaining, params.kv_bytes_per_token,
                           dst.prefill_speed, bw_eff)
        return d, 0.0, remaining, False


class AlwaysRecompute(Policy):
    name = "always-recompute"

    def plan_handover(self, user, dst, params, bw_eff):
        remaining = user.tokens
        d = handover_delay(remaining, 0.0, params.kv_bytes_per_token,
                           dst.prefill_speed, bw_eff)
        return d, remaining, 0.0, False


class ReactiveHybrid(Policy):
    name = "reactive-hybrid"

    def plan_handover(self, user, dst, params, bw_eff):
        remaining = user.tokens
        p = optimal_prefill_length(remaining, params.kv_bytes_per_token,
                                   dst.prefill_speed, bw_eff)
        d = handover_delay(p, remaining - p, params.kv_bytes_per_token,
                           dst.prefill_speed, bw_eff)
        return d, p, remaining - p, False


class PredictivePrefetch(ReactiveHybrid):
    """Proposed method: predict the next serving server and stream KV toward it
    ahead of the handover, adapt the recompute ratio under contention, and skip
    prefetch when a quick return (ping-pong) is predicted."""

    name = "predictive-prefetch"

    def __init__(self, horizon=45.0, prefetch_rate=250.0, pingpong_window=20.0):
        self.horizon = horizon              # seconds to extrapolate position
        self.prefetch_rate = prefetch_rate  # tokens/sec streamed ahead of handover
        self.pingpong_window = pingpong_window

    def _predict_target(self, user, servers):
        fx = user.x + user.vx * self.horizon
        fy = user.y + user.vy * self.horizon
        best, best_d = None, float("inf")
        for s in servers:
            d = distance(fx, fy, s.x, s.y)
            if d < best_d:
                best, best_d = s, d
        return best

    def _likely_return(self, user, target_id, t):
        for sid, ts in reversed(user.history):
            if t - ts > self.pingpong_window:
                break
            if sid == target_id:
                return True
        return False

    def on_step(self, t, dt, users, servers, params, gate_level):
        activations = 0
        for user in users:
            predicted = self._predict_target(user, servers)
            if predicted is None or predicted.id == user.server:
                continue
            # Ping-pong gate: skip prefetch toward a server we just left.
            if self._likely_return(user, predicted.id, t):
                continue
            # Safety gate: intervene proportionally to recent mobility/contention.
            if predicted.id != user.prefetch_target:
                user.prefetch_target = predicted.id
                user.prefetched = 0.0
            room = max(0.0, user.tokens - user.prefetched)
            step_prefetch = min(room, self.prefetch_rate * dt * gate_level)
            if step_prefetch > 0:
                user.prefetched += step_prefetch
                activations += 1
        return activations

    def plan_handover(self, user, dst, params, bw_eff):
        # Prefetch only helps if it was aimed at the actual target.
        prefetched = user.prefetched if user.prefetch_target == dst.id else 0.0
        remaining = max(0.0, user.tokens - prefetched)
        p = optimal_prefill_length(remaining, params.kv_bytes_per_token,
                                   dst.prefill_speed, bw_eff)
        d = handover_delay(p, remaining - p, params.kv_bytes_per_token,
                           dst.prefill_speed, bw_eff)
        used = prefetched > 0.0
        user.prefetched = 0.0
        user.prefetch_target = -1
        return d, p, remaining - p, used


def all_policies():
    return [AlwaysTransfer(), AlwaysRecompute(), ReactiveHybrid(), PredictivePrefetch()]
