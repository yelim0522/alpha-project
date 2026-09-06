"""Small-instance optimality gap of the coordinated trigger scheduler.

For K users predicted to the same target with heterogeneous contexts and
handover instants, enumerate every trigger schedule on a coarse time grid
(brute force), evaluate each with the mini simulator, and compare the best
schedule with what Coordinated and PallasApprox choose online.

Objective (Pallas's per-user cost summed over users):
    sum_i  alpha * SIT_i + (1 - alpha) * T_early_i          alpha = 0.8

The GPU is made scarce on purpose (prefill_parallel = 2) so that contention is
binding at K = 3..5; without contention every schedule that finishes on time
is optimal and the gap is trivially zero.

Run:  python3 optgap.py [--instances 8] [--k 3,4,5] [--grid 0.5]
"""

import argparse
import itertools
import random
from typing import List, Sequence

from minisim import mini_sim
from policies import Coordinated, PallasApprox

ALPHA = 0.8


class FixedSchedule(Coordinated):
    """Trigger user i at trig[i] (stream suffix), otherwise behave like Coordinated
    at the handover instant (residual recovery with prefix/suffix split)."""

    def __init__(self, trig: Sequence[float], grid: float = 0.25):
        super().__init__(detour=False, label="fixed", grid=grid)
        self.trig = list(trig)

    def on_step(self, t, dt, users, servers, params, predictions, loads, metrics):
        for u in users:
            pred = predictions.get(u.id)
            if pred is None or u.prep is not None:
                continue
            if self.trig[u.id] <= t + 1e-9 and self.trig[u.id] < pred.t_ho - 1e-9:
                self._trigger(u, pred.target, t, pred.t_ho, stream_suffix=True)


def objective(sits: List[float], earlies: List[float]) -> float:
    return sum(ALPHA * s + (1 - ALPHA) * e for s, e in zip(sits, earlies))


def evaluate(policy, inst, dt, parallel):
    sits, earlies = mini_sim(policy, inst["model"], inst["link"], inst["ctx"], inst["t_ho"],
                             dt, prefill_parallel=parallel, with_early=True)
    return objective(sits, earlies), sits


def brute_force(inst, grid, dt, parallel):
    options = []
    for t_ho in inst["t_ho"]:
        opts = [t_ho]                                  # "no preparation" (reactive)
        tw = grid
        while tw <= t_ho + 1e-9:
            opts.append(round(t_ho - tw, 6))
            tw += grid
        options.append(opts)
    best, best_sched, best_sits = float("inf"), None, None
    for sched in itertools.product(*options):
        j, sits = evaluate(FixedSchedule(sched), inst, dt, parallel)
        if j < best - 1e-12:
            best, best_sched, best_sits = j, sched, sits
    return best, best_sched, best_sits


def random_instance(rng, K, model="qwen32b", link=300.0):
    return dict(model=model, link=link,
                ctx=[rng.choice([1000, 2000, 3000, 4000]) for _ in range(K)],
                t_ho=[rng.choice([2.0, 2.5, 3.0, 3.5, 4.0]) for _ in range(K)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--k", type=str, default="3,4,5")
    ap.add_argument("--grid", type=float, default=0.25,
                    help="trigger grid shared by brute force and both online policies (s)")
    ap.add_argument("--dt", type=float, default=0.25, help="control period (= grid so that all "
                    "three decide on the same discrete instants)")
    ap.add_argument("--parallel", type=float, default=2.0, help="GPU batched prefill multiple")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    print(f"objective = sum_i 0.8*SIT_i + 0.2*T_early_i ; prefill_parallel={a.parallel}, grid={a.grid}s")
    print(f"  {'K':>2} {'inst':>4} | {'OPT':>7} {'Coord':>7} {'gap':>6} | {'Pallas':>7} {'gap':>6} | "
          f"{'meanSIT opt/coord/pallas':>26}")
    for K in [int(x) for x in a.k.split(",")]:
        gaps_c, gaps_p = [], []
        for i in range(a.instances):
            inst = random_instance(rng, K)
            grid = a.grid if K <= 4 else max(a.grid, 0.5)
            opt, _, opt_sits = brute_force(inst, grid, a.dt, a.parallel)
            jc, sc = evaluate(Coordinated(grid=a.grid), inst, a.dt, a.parallel)
            jp, sp = evaluate(PallasApprox(grid=a.grid), inst, a.dt, a.parallel)
            gc, gp = 100 * (jc - opt) / opt, 100 * (jp - opt) / opt
            gaps_c.append(gc)
            gaps_p.append(gp)
            print(f"  {K:>2} {i:>4} | {opt:7.3f} {jc:7.3f} {gc:+5.1f}% | {jp:7.3f} {gp:+5.1f}% | "
                  f"{sum(opt_sits) / K:8.3f} {sum(sc) / K:8.3f} {sum(sp) / K:8.3f}")
        print(f"  K={K}: mean gap  coordinated {sum(gaps_c) / len(gaps_c):+.1f}%  "
              f"(max {max(gaps_c):+.1f}%)   pallas {sum(gaps_p) / len(gaps_p):+.1f}%  "
              f"(max {max(gaps_p):+.1f}%)\n")


if __name__ == "__main__":
    main()
