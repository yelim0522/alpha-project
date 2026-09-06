"""Handover state-transfer policies.

Reactive baselines (state recovery starts at the handover instant):
  - AlwaysTransfer  : Full-Copy, transfer the whole KV cache (p = 0).
  - AlwaysRecompute : Recomputation, prefill the whole context (p = L).
  - ReactiveHybrid  : ctHO approximation, optimal prefix/suffix split p* at handover.
  - Detour          : keep the state at the source and forward tokens (no SIT,
                      persistent per-hop ITL penalty).
Proactive, uncoordinated (Pallas approximation):
  - PallasApprox    : per-user prefetching-window selection by grid search on
                      J = a*T_SIT + (1-a)*T_early using EWMA-smoothed *observed*
                      effective rates; prefix prefill (FCFS) + suffix streaming.
Proactive, coordinated (proposed):
  - Coordinated     : per-target scheduler that plans trigger times jointly for all
                      users predicted toward the same target (deadline-ordered,
                      capacity-capped staggering), chooses per-user suffix handling
                      (stream vs. defer-and-recompute) from planned link load,
                      serves prefill EDF, admits by VRAM budget, and decides
                      migrate-vs-detour for unplanned/ping-pong handovers.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from environment import (
    Server, User, Prep, CostParams, LinkLoad,
    optimal_prefill_length, handover_delay, residual_recovery, vram_in_use,
    fcfs_key, edf_key,
)
from prediction import Prediction


@dataclass
class Decision:
    sit: float
    recompute_tokens: float
    transfer_mb: float
    mode: str          # "migrate" | "detour" | "return"
    used_prep: bool


class Policy:
    name = "base"
    order_key = staticmethod(fcfs_key)   # GPU prefill service order at a target

    def reset(self):
        pass

    def on_step(self, t, dt, users, servers, params, predictions, loads, metrics):
        """Called every control cycle before resources are advanced."""

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong,
                      predicted, t) -> Decision:
        raise NotImplementedError

    # ----- shared helpers -------------------------------------------------- #
    @staticmethod
    def _reactive(user, dst, params, v_share, bw_share, how="hybrid") -> Decision:
        L, c = user.tokens, params.kv_mb_per_token
        if how == "transfer":
            p = 0.0
        elif how == "recompute":
            p = L
        else:
            p = optimal_prefill_length(L, c, v_share, bw_share)
        d = handover_delay(p, L - p, c, v_share, bw_share)
        return Decision(params.activation_latency + d, p, c * (L - p), "migrate", False)

    @staticmethod
    def _cancel(user, params, metrics, t):
        if user.prep is not None:
            metrics.record_cancel(user.prep, params.kv_mb_per_token)
            user.prep = None
            user.epoch += 1

    def _trigger(self, user, target, t, deadline, stream_suffix=True):
        user.epoch += 1
        user.prep = Prep(target=target, epoch=user.epoch, trigger_t=t, deadline_t=deadline,
                         prefix_total=user.tokens, prefix_remaining=user.tokens,
                         stream_suffix=stream_suffix)
        log = getattr(self, "trigger_log", None)
        if log is not None:
            log.append(t)

    @staticmethod
    def _residual(user, dst, params, v_share, bw_share, allow_split) -> Decision:
        prep, c = user.prep, params.kv_mb_per_token
        d, rec, xfer = residual_recovery(prep.prefix_remaining, prep.unsent_suffix_tokens(c),
                                         c, v_share, bw_share, allow_split)
        return Decision(params.activation_latency_proactive + d, rec, c * xfer, "migrate", True)


# --------------------------------------------------------------------------- #
# Reactive baselines
# --------------------------------------------------------------------------- #

class AlwaysTransfer(Policy):
    name = "full-copy"

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        return self._reactive(user, dst, params, v_share, bw_share, "transfer")


class AlwaysRecompute(Policy):
    name = "recompute"

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        return self._reactive(user, dst, params, v_share, bw_share, "recompute")


class ReactiveHybrid(Policy):
    name = "ctho-approx"

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        return self._reactive(user, dst, params, v_share, bw_share, "hybrid")


class Detour(Policy):
    name = "detour"

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        return Decision(0.0, 0.0, 0.0, "detour", False)


# --------------------------------------------------------------------------- #
# Proactive, uncoordinated (Pallas approximation)
# --------------------------------------------------------------------------- #

class PallasApprox(Policy):
    """Independent per-user prefetching-window selection (Pallas, Alg. 1)."""

    name = "pallas-approx"

    def __init__(self, alpha=0.8, t_max=5.0, grid=0.2, ewma=0.5, label=None):
        self.alpha = alpha
        self.t_max = t_max
        self.grid = grid
        self.ewma = ewma
        if label:
            self.name = label
        self.reset()

    def reset(self):
        self.obs_rp: Dict[int, float] = {}
        self.obs_bw: Dict[int, float] = {}

    def _observe(self, servers: Dict[int, Server], loads: Dict[int, LinkLoad]):
        for sid, srv in servers.items():
            ld = loads.get(sid) if loads else None
            rp = srv.prefill_share(ld.prefills if ld else 0)
            bw = srv.backhaul_bw / max(1, ld.streams if ld else 1)
            self.obs_rp[sid] = (1 - self.ewma) * self.obs_rp.get(sid, srv.prefill_speed) + self.ewma * rp
            self.obs_bw[sid] = (1 - self.ewma) * self.obs_bw.get(sid, srv.backhaul_bw) + self.ewma * bw

    def _cost(self, tw, t_remain, K, rp, bw, params):
        c, rd, T0 = params.kv_mb_per_token, params.decode_rate, params.activation_latency_proactive
        L_hist = K + rd * (t_remain - tw)
        T_hist = L_hist / rp
        S_inc = rd * tw * c
        T_res = max(0.0, S_inc / bw - tw)
        T_sit = T0 + max(0.0, T_hist - tw, T_res)
        T_early = max(0.0, tw - T_hist)
        return self.alpha * T_sit + (1 - self.alpha) * T_early

    def _windows(self, limit):
        """Candidate windows {0, grid, 2*grid, ..., limit} (limit always included)."""
        n = int(limit / self.grid + 1e-9)
        out = [i * self.grid for i in range(n + 1)]
        if limit - out[-1] > 1e-9:
            out.append(limit)
        return out

    def _select_window(self, t_remain, K, rp, bw, params):
        limit = min(self.t_max, t_remain)
        best_tw, best_j = 0.0, float("inf")
        for tw in self._windows(limit):
            j = self._cost(tw, t_remain, K, rp, bw, params)
            if j < best_j - 1e-12:
                best_tw, best_j = tw, j
        return best_tw

    @staticmethod
    def _fire_now(t_remain, tw, dt):
        """Pallas triggers when T_remain <= T_w*. With a discrete control period the
        ideal trigger instant t_remain - tw may fall between two cycles; firing at
        the last cycle before it (early by < dt) is preferred to firing late."""
        return t_remain - tw < dt - 1e-9

    def on_step(self, t, dt, users, servers, params, predictions, loads, metrics):
        self._observe(servers, loads)
        c = params.kv_mb_per_token
        for u in users:
            pred: Optional[Prediction] = predictions.get(u.id)
            if u.prep is not None:
                if pred is None or pred.target != u.prep.target:
                    self._cancel(u, params, metrics, t)     # target changed -> new epoch
                else:
                    u.prep.deadline_t = pred.t_ho           # time revision keeps state
                continue
            if pred is None or pred.target == u.anchor:
                continue
            t_remain = pred.t_ho - t
            if t_remain <= 0:
                continue
            tw = self._select_window(t_remain, u.tokens, self.obs_rp[pred.target],
                                     self.obs_bw[pred.target], params)
            if self._fire_now(t_remain, tw, dt):
                srv = servers[pred.target]
                if vram_in_use(users, pred.target, c) + u.tokens * c <= srv.vram_budget_mb:
                    self._trigger(u, pred.target, t, pred.t_ho, stream_suffix=True)

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        if user.prep is not None and user.prep.target == dst.id:
            return self._residual(user, dst, params, v_share, bw_share, allow_split=False)
        return self._reactive(user, dst, params, v_share, bw_share, "hybrid")


# --------------------------------------------------------------------------- #
# Proactive, coordinated (proposed)
# --------------------------------------------------------------------------- #

class Coordinated(PallasApprox):
    """Target-level joint planning of prefetching windows.

    For every target m and control cycle, users predicted toward m are planned in
    deadline order. Each user is given the window that minimises the Pallas
    objective *subject to* a cap on planned concurrency over that window, using
    nominal capacities and the planned (not observed) sharing factor k. Later
    arrivals are therefore pushed to earlier, emptier slots instead of piling
    onto the same instant. Suffix handling is chosen per user: stream only if the
    planned aggregate stream rate stays under `stream_util`, else defer and
    recompute/transfer the (short) suffix at handover with an optimal split.

    With a single candidate and no active preparation (k = 1) the evaluation
    reduces exactly to Pallas's grid search: same window grid, same objective,
    same trigger rule, suffix always streamed. Coordination only changes the
    decision when it can see contention.

    Ablation switches (all True = proposed method):
      planned_k   : rate estimate from planned occupancy instead of observed EWMA
      k_max       : concurrency cap that forces staggering (None = uncapped)
      suffix_mode : per-user stream-vs-defer choice (False = always stream)
      edf         : deadline-ordered prefill service at the target (False = FCFS)
      detour      : transient detour + background settle for unplanned/ping-pong
                    handovers (False = reactive hybrid like Pallas's fallback)
    """

    name = "coordinated"

    def __init__(self, alpha=0.8, t_max=15.0, grid=0.2, k_max=4, stream_util=0.6,
                 margin_min=0.0, detour_hops=2, settle_grace=10.0, label=None,
                 planned_k=True, suffix_mode=True, edf=True, detour=True, occ_mode="gpu"):
        super().__init__(alpha=alpha, t_max=t_max, grid=grid, label=label)
        self.k_max = k_max if k_max is not None else 10 ** 9
        self.occ_mode = occ_mode          # "gpu": prefill span only; "window": whole window
        self.stream_util = stream_util
        self.margin_min = margin_min
        self.detour_hops = detour_hops if detour else 0
        self.settle_grace = settle_grace
        self.planned_k = planned_k
        self.suffix_mode = suffix_mode
        self.order_key = edf_key if edf else fcfs_key

    def _cost_k(self, tw, t_remain, K, srv, k, params, contended):
        """Pallas objective evaluated with planned sharing factor k; returns
        (J, stream_suffix)."""
        c, rd, T0 = params.kv_mb_per_token, params.decode_rate, params.activation_latency_proactive
        if self.planned_k:
            rp, bw = srv.prefill_share(k), srv.backhaul_bw / k
        else:
            rp, bw = self.obs_rp[srv.id], self.obs_bw[srv.id]
        L_hist = K + rd * (t_remain - tw)
        T_hist = L_hist / rp
        T_res_stream = max(0.0, rd * tw * c / bw - tw)
        stream = True
        T_res = T_res_stream
        if self.suffix_mode and contended:
            stream_ok = k * rd * c <= self.stream_util * srv.backhaul_bw
            T_res_defer = handover_delay(*_split_suffix(rd * tw, c, rp, bw), c, rp, bw)
            if not stream_ok or T_res_defer < T_res_stream:
                T_res, stream = T_res_defer, False
        T_sit = T0 + max(0.0, T_hist - tw, T_res)
        T_early = max(0.0, tw - T_hist)
        return self.alpha * T_sit + (1 - self.alpha) * T_early, stream

    def reset(self):
        super().reset()
        # uid -> dict(target, t_ho, start, tw, dur, stream): a plan is sticky until the
        # prediction changes, mirroring Pallas's one-time trigger decision.
        self.plans: Dict[int, dict] = {}

    def _plan_target(self, srv, cands, active, kept, t, dt, params):
        """Plan the unplanned candidates `cands` around `active` preparations and
        `kept` plans. Returns (new_plans, triggers) where triggers is
        [(user, stream_suffix)] for plans whose start falls in this cycle."""
        g = self.grid
        contended = len(cands) + len(active) + len(kept) > 1
        horizon = max([p.t_ho for _, p in cands] + [pl["t_ho"] for pl in kept]) - t
        # Extend past the last deadline so that residual (post-handover) prefill of
        # short-window plans is visible as GPU occupancy to later plans.
        nbins = max(2, int((horizon + self.t_max) / g) + 2)
        # GPU occupancy: bins in which a prefix prefill is expected to be running.
        # Streaming-only preparations (prefix done) do not occupy the GPU.
        occ = [0] * nbins
        v1 = srv.prefill_speed

        def span(start_bin, dur_s):
            # half-open bin range covering [start, start + dur)
            return max(0, start_bin), min(nbins, start_bin + max(1, math.ceil(dur_s / g - 1e-9)))

        def occupy(start_bin, dur_s):
            lo, hi = span(start_bin, dur_s)
            for b in range(lo, hi):
                occ[b] += 1

        for prep in active:
            if self.occ_mode == "window":
                occupy(0, max(0.0, prep.deadline_t - t))
            elif prep.prefix_remaining > 0:
                occupy(0, prep.prefix_remaining / v1)
        for pl in kept:
            occupy(int((pl["start"] - t) / g), pl["dur"] if self.occ_mode != "window" else pl["tw"])

        def window_slot(tw, t_remain, K):
            s_bin = max(0, int((t_remain - tw) / g))
            L_hist = K + params.decode_rate * (t_remain - tw)
            dur = tw if self.occ_mode == "window" else L_hist / v1
            lo, hi = span(s_bin, dur)
            k = 1 + max(occ[lo:hi]) if contended and hi > lo else 1
            return s_bin, dur, k

        new_plans, triggers = {}, []
        for u, pred in sorted(cands, key=lambda up: up[1].t_ho):
            t_remain = pred.t_ho - t
            if t_remain <= 0:
                continue
            limit = min(self.t_max, t_remain)
            best = None   # (J, tw, start_bin, prefill_dur, stream)
            for capped in (True, False):
                for tw in self._windows(limit):
                    s_bin, dur, k = window_slot(tw, t_remain, u.tokens)
                    if not capped or k <= self.k_max:
                        j, stream = self._cost_k(tw, t_remain, u.tokens, srv, k, params, contended)
                        if best is None or j < best[0] - 1e-12:
                            best = (j, tw, s_bin, dur, stream)
                if best is not None:
                    break                    # no slot under the cap -> uncapped fallback
            _, tw, s_bin, dur, stream = best
            occupy(s_bin, dur)
            new_plans[u.id] = dict(target=srv.id, t_ho=pred.t_ho, start=pred.t_ho - tw,
                                   tw=tw, dur=dur, stream=stream)
            if self._fire_now(t_remain, tw, dt):   # same trigger rule as Pallas
                triggers.append((u, stream))
        return new_plans, triggers

    def on_step(self, t, dt, users, servers, params, predictions, loads, metrics):
        self._observe(servers, loads)
        c = params.kv_mb_per_token
        cands: Dict[int, list] = {}
        kept: Dict[int, list] = {}
        active: Dict[int, list] = {}
        tol = max(self.grid, dt)
        for u in users:
            pred = predictions.get(u.id)
            if u.prep is not None:
                self.plans.pop(u.id, None)
                if u.prep.target == u.server:
                    continue                                  # settle migration in flight
                if pred is None or pred.target != u.prep.target:
                    self._cancel(u, params, metrics, t)
                    continue
                u.prep.deadline_t = pred.t_ho
                active.setdefault(u.prep.target, []).append(u.prep)
                continue
            # Detoured user that has stayed: migrate state to the serving server
            # in the background (no handover deadline; short window).
            if u.hops > 0 and u.history and t - u.history[-1][1] >= self.settle_grace:
                self.plans.pop(u.id, None)
                if vram_in_use(users, u.server, c) + u.tokens * c <= servers[u.server].vram_budget_mb:
                    self._trigger(u, u.server, t, t + 2.0, stream_suffix=False)
                continue
            if pred is None or pred.target == u.anchor:
                self.plans.pop(u.id, None)
                continue
            pl = self.plans.get(u.id)
            if pl is not None and pl["target"] == pred.target and abs(pl["t_ho"] - pred.t_ho) <= tol:
                kept.setdefault(pred.target, []).append((u, pl))
            else:
                self.plans.pop(u.id, None)
                cands.setdefault(pred.target, []).append((u, pred))

        for sid in set(cands) | set(kept):
            srv = servers[sid]
            lst = cands.get(sid, [])
            kept_here = kept.get(sid, [])
            if self.margin_min > 0 and len(lst) + len(kept_here) + len(active.get(sid, [])) > 1:
                # Optional confidence gate; only when preparations compete for the target.
                lst = [(u, p) for u, p in lst if p.margin >= self.margin_min]
            triggers = []
            for u, pl in kept_here:
                if pl["start"] - t < dt - 1e-9:
                    triggers.append((u, pl["stream"]))
            if lst:
                new_plans, new_trig = self._plan_target(srv, lst, active.get(sid, []),
                                                        [pl for _, pl in kept_here], t, dt, params)
                self.plans.update(new_plans)
                triggers += new_trig
            for u, stream in triggers:
                if vram_in_use(users, sid, c) + u.tokens * c <= srv.vram_budget_mb:
                    self.plans.pop(u.id, None)
                    self._trigger(u, sid, t, predictions[u.id].t_ho, stream_suffix=stream)

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        if user.prep is not None and user.prep.target == dst.id:
            return self._residual(user, dst, params, v_share, bw_share, allow_split=True)
        unplanned = predicted is None or predicted.target != dst.id
        if (is_pingpong or unplanned) and user.hops < self.detour_hops:
            return Decision(0.0, 0.0, 0.0, "detour", False)
        return self._reactive(user, dst, params, v_share, bw_share, "hybrid")


def ablation_ladder():
    """Coordinated variants adding one mechanism at a time. 'abl-+edf' is the full
    method without detour; 'abl-+detour' equals `coordinated`."""
    return [
        Coordinated(label="abl-planned-k", k_max=None, suffix_mode=False, edf=False, detour=False),
        Coordinated(label="abl-+cap", suffix_mode=False, edf=False, detour=False),
        Coordinated(label="abl-+suffix", edf=False, detour=False),
        Coordinated(label="abl-+edf", detour=False),
        Coordinated(label="abl-+detour"),
    ]


def _split_suffix(tokens, c, v, bw):
    p = optimal_prefill_length(tokens, c, v, bw)
    return p, tokens - p


def all_policies():
    return [AlwaysTransfer(), AlwaysRecompute(), ReactiveHybrid(), Detour(),
            PallasApprox(), Coordinated()]
