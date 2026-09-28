"""Characterize the current planner, including limits (not new guarantees).

These tests freeze observed behavior for the algorithm audit. Tests describing
limits must be updated deliberately if a future implementation fixes the limit.
"""

import unittest

from environment import Server, User, CostParams, Prep, advance_preparations, vram_in_use
from metrics import Metrics
from policies import Coordinated, PallasApprox, ablation_ladder
from prediction import Prediction


class CoordinationContractTests(unittest.TestCase):
    def setUp(self):
        self.srv = Server(1, 1, 0, 100, 1970, 37.5, 6000, 4)
        self.params = CostParams()

    def user(self, uid=0, tokens=1000):
        return User(uid, 0, 0, tokens=tokens, server=0, anchor=0)

    def cycle(self, policy, users, predictions, t=0, dt=0.5):
        policy.on_step(t, dt, users, {1: self.srv}, self.params,
                       predictions, {}, Metrics())

    def test_empty_profile_matches_fresh_pallas_cost(self):
        p, c = PallasApprox(), Coordinated()
        for tw in p._windows(4):
            expected = p._cost(tw, 4, 1000, self.srv.prefill_speed,
                               self.srv.backhaul_bw, self.params)
            actual, stream, _ = c._cost_k(tw, 4, 1000, self.srv,
                                        ([0] * 100, [0] * 100), 0, self.params, False)
            self.assertAlmostEqual(actual, expected)
            self.assertTrue(stream)

    def test_deadline_is_soft_not_a_feasibility_constraint(self):
        c, u = Coordinated(), self.user(tokens=100000)
        plans, _ = c._plan_target(self.srv, [(u, Prediction(1, 1, 1))],
                                   [], [], 0, 0.5, self.params)
        plan = plans[u.id]
        self.assertGreater(plan['start'] + plan['dur'], plan['t_ho'])

    def test_small_prediction_change_keeps_old_plan(self):
        c, u = Coordinated(), self.user()
        self.cycle(c, [u], {0: Prediction(1, 10, 1)})
        old = c.plans[0].copy()
        self.cycle(c, [u], {0: Prediction(1, 10.1, 1)}, t=0.5)
        self.assertEqual(c.plans[0], old)

    def test_large_prediction_change_replans(self):
        c, u = Coordinated(), self.user()
        self.cycle(c, [u], {0: Prediction(1, 10, 1)})
        self.cycle(c, [u], {0: Prediction(1, 11, 1)}, t=0.5)
        self.assertEqual(c.plans[0]['t_ho'], 11)

    def test_same_deadline_has_no_edf_externality_penalty(self):
        c = Coordinated()
        self.srv.prefill_parallel = 1
        # Occupancy of earlier/equal deadlines blocks us, but ext counts only
        # displaced later-deadline work under EDF.
        duration, externality, _ = c._fluid_prefill([1] * 5, [0] * 5,
                                                   0, 100, self.srv, True)
        self.assertGreater(duration, 1)
        self.assertEqual(externality, 0)
        _, externality, _ = c._fluid_prefill([0] * 5, [1] * 5,
                                            0, 100, self.srv, True)
        self.assertGreater(externality, 0)

    def test_vram_admission_can_defer_a_due_plan(self):
        c, u = Coordinated(), self.user()
        self.srv.vram_budget_mb = 0
        self.cycle(c, [u], {0: Prediction(1, 0.2, 1)})
        self.assertIsNone(u.prep)
        self.assertIn(0, c.plans)

    def test_vram_gate_does_not_reserve_unbuilt_prefixes(self):
        c, users = Coordinated(), [self.user(0), self.user(1)]
        self.srv.vram_budget_mb = 1000 * self.params.kv_mb_per_token
        self.cycle(c, users, {u.id: Prediction(1, 0.2, 1) for u in users})
        self.assertTrue(all(u.prep is not None for u in users))
        # Both admissions see zero materialized prefix. The model has no later
        # memory enforcement, so this is not a global VRAM-cap guarantee.
        advance_preparations(users, {1: self.srv}, self.params, 0, 1, c.order_key)
        self.assertGreater(vram_in_use(users, 1, self.params.kv_mb_per_token),
                           self.srv.vram_budget_mb)

    def test_suffix_switch_does_not_disable_residual_split(self):
        c, p, u = Coordinated(suffix_mode=False, detour=False), PallasApprox(), self.user()
        u.prep = Prep(1, 1, 0, 1, 1000, 0, suffix_backlog_mb=100)
        c_dec = c.plan_handover(u, self.srv, self.params, 100, 1, False, None, 1)
        p_dec = p.plan_handover(u, self.srv, self.params, 100, 1, False, None, 1)
        self.assertLess(c_dec.sit, p_dec.sit)

    def test_existing_ladder_is_not_timing_only(self):
        first = ablation_ladder()[0]
        self.assertTrue(first.planned_k)
        self.assertFalse(first.social)
        self.assertFalse(first.suffix_mode)
        self.assertEqual(first.detour_hops, 0)
        self.assertIs(first.plan_handover.__func__, Coordinated.plan_handover)


if __name__ == '__main__':
    unittest.main()
