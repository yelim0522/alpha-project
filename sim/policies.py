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

    @staticmethod
    def _trigger(user, target, t, deadline, stream_suffix=True):
        user.epoch += 1
        user.prep = Prep(target=target, epoch=user.epoch, trigger_t=t, deadline_t=deadline,
                         prefix_total=user.tokens, prefix_remaining=user.tokens,
                         stream_suffix=stream_suffix)

    @staticmethod
    def _residual(user, dst, params, v_share, bw_share, allow_split) -> Decision:
        prep, c = user.prep, params.kv_mb_per_token
        d, rec, xfer = residual_recovery(prep.prefix_remaining, prep.unsent_suffix_tokens(c),
                                         c, v_share, bw_share, allow_split)
        return Decision(params.activation_latency + d, rec, c * xfer, "migrate", True)


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
            rp = srv.prefill_speed / max(1, ld.prefills if ld else 1)
            bw = srv.backhaul_bw / max(1, ld.streams if ld else 1)
            self.obs_rp[sid] = (1 - self.ewma) * self.obs_rp.get(sid, srv.prefill_speed) + self.ewma * rp
            self.obs_bw[sid] = (1 - self.ewma) * self.obs_bw.get(sid, srv.backhaul_bw) + self.ewma * bw

    def _cost(self, tw, t_remain, K, rp, bw, params):
        c, rd, T0 = params.kv_mb_per_token, params.decode_rate, params.activation_latency
        L_hist = K + rd * (t_remain - tw)
        T_hist = L_hist / rp
        S_inc = rd * tw * c
        T_res = max(0.0, S_inc / bw - tw)
        T_sit = T0 + max(0.0, T_hist - tw, T_res)
        T_early = max(0.0, tw - T_hist)
        return self.alpha * T_sit + (1 - self.alpha) * T_early

    def _select_window(self, t_remain, K, rp, bw, params):
        limit = min(self.t_max, t_remain)
        best_tw, best_j = 0.0, float("inf")
        tw = 0.0
        while tw <= limit + 1e-9:
            j = self._cost(tw, t_remain, K, rp, bw, params)
            if j < best_j:
                best_tw, best_j = tw, j
            tw += self.grid
        return best_tw

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
            if t_remain <= tw + 1e-9:
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
    deadline order. Each user is given the trigger time that minimises the Pallas
    objective *subject to* a cap on planned concurrency over its window, using
    nominal capacities and the planned (not observed) sharing factor. Later
    arrivals are therefore pushed to earlier, emptier slots instead of piling
    onto the same instant. Suffix handling is chosen per user: stream only if the
    planned aggregate stream rate stays under `stream_util`, else defer and
    recompute/transfer the (short) suffix at handover with an optimal split."""

    name = "coordinated"
    order_key = staticmethod(edf_key)

    def __init__(self, alpha=0.8, t_max=15.0, grid=0.5, k_max=4, stream_util=0.6,
                 margin_min=0.05, detour_hops=2, settle_grace=10.0, label=None):
        super().__init__(alpha=alpha, t_max=t_max, grid=grid, label=label)
        self.k_max = k_max
        self.stream_util = stream_util
        self.margin_min = margin_min
        self.detour_hops = detour_hops
        self.settle_grace = settle_grace

    def _cost_k(self, tw, t_remain, K, srv, k, params):
        """Pallas objective evaluated with planned sharing factor k; returns
        (J, stream_suffix)."""
        c, rd, T0 = params.kv_mb_per_token, params.decode_rate, params.activation_latency
        rp, bw = srv.prefill_speed / k, srv.backhaul_bw / k
        L_hist = K + rd * (t_remain - tw)
        T_hist = L_hist / rp
        stream_ok = k * rd * c <= self.stream_util * srv.backhaul_bw
        T_res_stream = max(0.0, rd * tw * c / bw - tw)
        T_res_defer = handover_delay(*_split_suffix(rd * tw, c, rp, bw), c, rp, bw)
        if stream_ok and T_res_stream <= T_res_defer:
            T_res, stream = T_res_stream, True
        else:
            T_res, stream = T_res_defer, False
        T_sit = T0 + max(0.0, T_hist - tw, T_res)
        T_early = max(0.0, tw - T_hist)
        return self.alpha * T_sit + (1 - self.alpha) * T_early, stream

    def _plan_target(self, srv, cands, active, t, dt, params):
        """Return [(user, stream_suffix)] to trigger now at target srv."""
        if not cands:
            return []
        horizon = max(p.t_ho for _, p in cands) - t
        nbins = int(horizon / dt) + 2
        occ = [0] * nbins
        for prep in active:
            end = min(nbins - 1, max(0, int((prep.deadline_t - t) / dt)))
            for b in range(0, end + 1):
                occ[b] += 1
        triggers = []
        for u, pred in sorted(cands, key=lambda up: up[1].t_ho):
            t_remain = pred.t_ho - t
            dl_bin = min(nbins - 1, int(t_remain / dt))
            best = None   # (J, start_bin, stream)
            for s_bin in range(dl_bin, -1, -1):
                tw = t_remain - s_bin * dt
                if tw > self.t_max:
                    break
                k = 1 + max(occ[s_bin:dl_bin + 1])
                if k > self.k_max:
                    continue
                j, stream = self._cost_k(tw, t_remain, u.tokens, srv, k, params)
                if best is None or j < best[0] - 1e-9:
                    best = (j, s_bin, stream)
            if best is None:
                continue                      # no capacity within t_max: defer
            _, s_bin, stream = best
            for b in range(s_bin, dl_bin + 1):
                occ[b] += 1
            if s_bin == 0:
                triggers.append((u, stream))
        return triggers

    def on_step(self, t, dt, users, servers, params, predictions, loads, metrics):
        c = params.kv_mb_per_token
        cands: Dict[int, list] = {}
        active: Dict[int, list] = {}
        for u in users:
            pred = predictions.get(u.id)
            if u.prep is not None:
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
                if vram_in_use(users, u.server, c) + u.tokens * c <= servers[u.server].vram_budget_mb:
                    self._trigger(u, u.server, t, t + 2.0, stream_suffix=False)
                continue
            if pred is None or pred.target == u.anchor or pred.margin < self.margin_min:
                continue
            cands.setdefault(pred.target, []).append((u, pred))

        for sid, lst in cands.items():
            srv = servers[sid]
            for u, stream in self._plan_target(srv, lst, active.get(sid, []), t, dt, params):
                if vram_in_use(users, sid, c) + u.tokens * c <= srv.vram_budget_mb:
                    self._trigger(u, sid, t, predictions[u.id].t_ho, stream_suffix=stream)

    def plan_handover(self, user, dst, params, v_share, bw_share, is_pingpong, predicted, t):
        if user.prep is not None and user.prep.target == dst.id:
            return self._residual(user, dst, params, v_share, bw_share, allow_split=True)
        unplanned = predicted is None or predicted.target != dst.id
        if (is_pingpong or unplanned) and user.hops < self.detour_hops:
            return Decision(0.0, 0.0, 0.0, "detour", False)
        return self._reactive(user, dst, params, v_share, bw_share, "hybrid")


def _split_suffix(tokens, c, v, bw):
    p = optimal_prefill_length(tokens, c, v, bw)
    return p, tokens - p


def all_policies():
    return [AlwaysTransfer(), AlwaysRecompute(), ReactiveHybrid(), Detour(),
            PallasApprox(), Coordinated()]
