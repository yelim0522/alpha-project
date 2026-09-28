"""Policy-neutral preparation-KV reservations (not total physical GPU VRAM).

Opt-in reserved-v1: release detached owners, renew live work, admit due starts,
then progress resources. Reservations include materialized bytes, not add to them.
"""

from collections import Counter
from dataclasses import dataclass
import math

from environment import Prep


@dataclass
class StartRequest:
    policy: object
    user: object
    target: int
    t: float
    deadline: float
    tokens: float
    stream: bool
    copy: bool
    planned: float
    epoch: int


class PreparationMemory:
    version = "reserved-v1"

    def __init__(self, servers, params, metrics):
        self.servers, self.params, self.metrics = servers, params, metrics
        for s in servers.values():
            if not math.isfinite(s.vram_budget_mb) or s.vram_budget_mb < 0:
                raise ValueError("preparation memory budget must be finite and non-negative")
        if not math.isfinite(params.kv_mb_per_token) or params.kv_mb_per_token < 0:
            raise ValueError("KV size must be finite and non-negative")
        if not math.isfinite(params.decode_rate) or params.decode_rate < 0:
            raise ValueError("decode rate must be finite and non-negative")
        self.entries = {}  # owner -> (user, prep, reserved MB)
        self.pending = []
        self.deferred = set()
        self.stats = Counter()
        self.events = []  # audit trail, including planned and actual trigger times

    @staticmethod
    def preps(users):
        for u in users:
            for p in ([u.prep] if u.prep is not None else []) + list(u.hedge_preps.values()):
                yield u, p

    @staticmethod
    def owner(u, p):
        return (u.id, u.session, p.epoch, p.target)

    def total(self, target):
        return sum(r for owner, (_, _, r) in self.entries.items() if owner[-1] == target)

    def release(self, owner):
        if self.entries.pop(owner, None) is not None:
            self.stats['releases'] += 1

    def sync(self, users):
        live = {self.owner(u, p): p for u, p in self.preps(users)}
        for owner in list(self.entries):
            if owner not in live:
                self.release(owner)
        self.check(users)

    def request(self, policy, user, target, t, deadline, stream=True, copy=False,
                planned=None):
        # Capture prefix before the closed-loop source generates this step.
        self.pending.append(StartRequest(policy, user, target, t, deadline, user.tokens,
                                         stream, copy, t if planned is None else planned,
                                         user.epoch + 1))

    def _generated(self, user, dt):
        return (user.generated_tokens_step if user.generated_tokens_step >= 0
                else self.params.decode_rate * dt)

    def _required(self, prefix, suffix, deadline, t, dt, generated):
        forecast = self.params.decode_rate * (max(deadline - t, 0.0) + dt)
        return self.params.kv_mb_per_token * (prefix + suffix + max(forecast, generated))

    def _fits(self, target, extra):
        return self.total(target) + extra <= self.servers[target].vram_budget_mb

    def before_advance(self, users, t, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("reservation dt must be finite and positive")
        self.sync(users)
        # Renewals have priority over all new starts; no policy-specific priority.
        live = sorted(self.preps(users), key=lambda up:
                      (up[1].deadline_t, up[0].id, up[1].epoch, up[1].target))
        for u, p in live:
            owner = self.owner(u, p)
            old = self.entries[owner][2]
            required = max(old, self._required(p.prefix_total, p.suffix_tokens,
                           p.deadline_t, t, dt, self._generated(u, dt)))
            if not self._fits(p.target, required - old):
                self.stats['renew_rejected'] += 1
                self.stats['cancels'] += 1
                self.metrics.record_cancel(p, self.params.kv_mb_per_token)
                if u.prep is p:
                    u.prep = None
                else:
                    del u.hedge_preps[p.target]
                u.epoch += 1
                self.release(owner)
                self.events.append(dict(event='renew-cancel', t=t, owner=owner,
                                        requested_mb=required))
            elif required > old:
                self.entries[owner] = (u, p, required)
                self.stats['renewals'] += 1

        still_deferred = set()
        requests, self.pending = self.pending, []
        for req in sorted(requests, key=lambda r: (r.deadline, r.user.id, r.epoch, r.target)):
            u = req.user
            key = (u.id, u.session, req.target, req.copy)
            required = self._required(req.tokens, 0, req.deadline, t, dt,
                                      self._generated(u, dt))
            self.stats['start_attempts'] += 1
            if not self._fits(req.target, required):
                self.stats['defer_attempts'] += 1
                if key not in self.deferred and key not in still_deferred:
                    self.stats['deferred_requests'] += 1
                still_deferred.add(key)
                reason = ('oversize' if required > self.servers[req.target].vram_budget_mb
                          else 'capacity')
                self.stats[reason + '_attempts'] += 1
                self.events.append(dict(event='defer', reason=reason, t=t,
                                        user=u.id, session=u.session, target=req.target,
                                        planned_t=req.planned, requested_mb=required))
                continue
            owner = (u.id, u.session, u.epoch + 1, req.target)
            # Register before object creation; roll back if construction fails.
            self.entries[owner] = (u, None, required)
            try:
                p = Prep(req.target, owner[2], t, req.deadline, req.tokens, req.tokens,
                         stream_suffix=req.stream)
            except Exception:
                self.release(owner)
                raise
            u.epoch = p.epoch
            self.entries[owner] = (u, p, required)
            if req.copy:
                u.hedge_preps[req.target] = p
                self.metrics.record_hedge_copy()
            else:
                u.prep = p
                if hasattr(req.policy, 'plans'):
                    req.policy.plans.pop(u.id, None)
                if getattr(req.policy, 'trigger_log', None) is not None:
                    req.policy.trigger_log.append(t)
            self.stats['admissions'] += 1
            self.stats['start_lag_sum_s'] += t - req.planned
            self.events.append(dict(event='admit', owner=owner, planned_t=req.planned,
                                    actual_t=t, reserved_mb=required))
        # A deferred request is one contiguous episode of due attempts for a
        # (user, session, target, copy). Stopped/replanned requests end the episode.
        self.deferred = still_deferred
        self.check(users)

    def check(self, users):
        self.stats['checks'] += 1
        c = self.params.kv_mb_per_token
        live = {self.owner(u, p): p for u, p in self.preps(users)}
        valid = set(live) == set(self.entries)
        actual = Counter()
        for owner, (_, p, reserved) in self.entries.items():
            a = p.vram_mb(c)
            valid &= live.get(owner) is p and -1e-8 <= a <= reserved + 1e-8
            actual[p.target] += a
        for sid, s in self.servers.items():
            r = self.total(sid)
            valid &= math.isfinite(r) and -1e-8 <= r <= s.vram_budget_mb + 1e-8
            self.stats['peak_reserved_target_mb'] = max(self.stats['peak_reserved_target_mb'], r)
            self.stats['peak_actual_target_mb'] = max(self.stats['peak_actual_target_mb'], actual[sid])
            fraction = r / s.vram_budget_mb if s.vram_budget_mb else 0.0
            self.stats['peak_budget_fraction'] = max(self.stats['peak_budget_fraction'], fraction)
        if not valid:
            self.stats['violations'] += 1
            raise AssertionError("preparation memory invariant violated")

    def after_advance(self, users, dt):
        self.check(users)
        # End-of-step rectangle integral across targets, MB*s (not exact area).
        self.stats['unused_reserved_mb_s'] += dt * sum(
            r - p.vram_mb(self.params.kv_mb_per_token) for _, p, r in self.entries.values())

    def finish(self, users):
        self.sync(users)
        self.stats['end_pending'] = len(self.entries)
        self.stats['end_reserved_mb'] = sum(r for _, _, r in self.entries.values())
        self.stats['end_actual_mb'] = sum(p.vram_mb(self.params.kv_mb_per_token)
                                         for _, p, _ in self.entries.values())
        for u in users:
            u.prep = None
            u.hedge_preps.clear()
        self.sync(users)
        self.pending.clear()
        self.deferred.clear()

    def summary(self):
        keys = ('start_attempts admissions defer_attempts deferred_requests oversize_attempts '
                'capacity_attempts renewals renew_rejected cancels releases checks violations '
                'peak_reserved_target_mb peak_actual_target_mb peak_budget_fraction '
                'unused_reserved_mb_s end_pending end_reserved_mb end_actual_mb').split()
        result = {'memory_' + k: self.stats[k] for k in keys}
        result['memory_start_lag_mean_s'] = (self.stats['start_lag_sum_s'] / self.stats['admissions']
                                            if self.stats['admissions'] else 0.0)
        return result


def configure_memory(policy, mode, servers, params, metrics):
    # Explicitly clear a previous run's manager even when reusing a policy.
    policy.memory = None
    if mode not in ('legacy', 'reserved'):
        raise ValueError('unknown memory mode: ' + mode)
    if mode == 'reserved':
        from policies import TurnBoundary
        if isinstance(policy, TurnBoundary):
            raise ValueError('TurnBoundary uses BoundaryJob; reserved mode is not supported')
        policy.memory = PreparationMemory(servers, params, metrics)
    return policy.memory
