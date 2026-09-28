"""Reserved-v1 invariants and lifecycle; legacy contracts remain separate."""

import random
import unittest
from unittest.mock import patch

from environment import Server, User, Prep, CostParams, advance_preparations
from memory import configure_memory
from metrics import Metrics
from minisim import mini_sim
from policies import PallasApprox, Coordinated, HedgedPallas, TurnBoundary
from prediction import Prediction
from run import build_parser, finalize_cfg, build_servers, precompute_trace, run_policy, policies_for


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.params = CostParams(kv_mb_per_token=1, decode_rate=10)
        self.servers = {sid: Server(sid, sid, 0, 100, 1000, 1000, 115)
                        for sid in (1, 2)}
        self.metrics = Metrics()
        self.policy = PallasApprox()
        self.memory = configure_memory(self.policy, 'reserved', self.servers,
                                       self.params, self.metrics)

    def user(self, uid=0, tokens=100):
        return User(uid, 0, 0, tokens=tokens, server=0, anchor=0)

    def progress(self, users, t=0, dt=0.5):
        return advance_preparations(users, self.servers, self.params, t, dt,
                                    self.policy.order_key, self.memory)

    def start(self, u, deadline=1, stream=True):
        self.policy._trigger(u, 1, 0, deadline, stream)
        self.progress([u])

    def test_same_cycle_reserves_unbuilt_prefix_and_orders_users(self):
        users = [self.user(1), self.user(0)]
        for u in users:
            self.policy._trigger(u, 1, 0, 1)
        self.assertTrue(all(u.prep is None for u in users))
        self.progress(users)
        self.assertIsNone(users[0].prep)
        self.assertIsNotNone(users[1].prep)
        self.assertEqual(self.memory.total(1), 115)
        self.assertEqual(self.memory.stats['defer_attempts'], 1)

    def test_deadline_priority_and_skip_oversize(self):
        users = [self.user(0, 500), self.user(1), self.user(2)]
        for u, d in zip(users, [0.1, 1, 0.5]):
            self.policy._trigger(u, 1, 0, d)
        self.progress(users)
        self.assertIsNotNone(users[2].prep)
        self.assertIsNone(users[1].prep)
        self.assertEqual(self.memory.stats['oversize_attempts'], 1)

    def test_zero_exact_and_oversize_budgets(self):
        for budget, accepted in [(0, False), (114.999, False), (115, True)]:
            with self.subTest(budget=budget):
                self.setUp()
                self.servers[1].vram_budget_mb = budget
                u = self.user()
                self.start(u)
                self.assertEqual(u.prep is not None, accepted)

    def test_stream_and_defer_reserve_same_amount(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.setUp()
                u = self.user()
                self.start(u, stream=stream)
                self.assertEqual(self.memory.total(1), 115)
                self.assertEqual(u.prep.vram_mb(1), 105 if stream else 100)

    def test_prefix_done_keeps_reservation_and_never_shrinks(self):
        u = self.user()
        self.start(u)
        self.assertEqual(u.prep.prefix_remaining, 0)
        u.prep.deadline_t = 0.5
        self.progress([u], t=0.5)
        self.assertEqual(self.memory.total(1), 115)

    def test_prediction_delay_expands_when_space_available(self):
        u = self.user()
        self.start(u)
        self.servers[1].vram_budget_mb = 200
        u.prep.deadline_t = 5
        self.progress([u], t=0.5)
        self.assertEqual(self.memory.total(1), 155)
        self.assertEqual(self.memory.stats['renewals'], 1)

    def test_failed_renewal_cancels_before_writes_counts_actual_waste(self):
        u = self.user()
        self.start(u)
        p, source_tokens = u.prep, u.tokens
        u.prep.deadline_t = 5
        self.progress([u], t=0.5)
        self.assertIsNone(u.prep)
        self.assertEqual(p.suffix_tokens, 5)
        self.assertEqual(self.metrics.wasted_mb, 105)  # not 115 reserved MB
        self.assertEqual(u.tokens, source_tokens)
        self.assertEqual(self.memory.total(1), 0)
        self.assertEqual(self.memory.stats['cancels'], 1)

    def test_actual_token_jump_guard_even_without_deadline_change(self):
        for stream in (False, True):
            self.setUp()
            u = self.user()
            self.start(u, stream=stream)
            p = u.prep
            u.generated_tokens_step = 50
            self.progress([u], t=0.5)
            self.assertIsNone(u.prep)
            self.assertEqual(p.suffix_tokens, 5)

    def test_new_start_also_reserves_actual_generation_burst(self):
        u = self.user()
        u.generated_tokens_step = 50
        self.start(u)
        self.assertIsNone(u.prep)
        self.assertEqual(self.memory.stats['oversize_attempts'], 1)

    def test_renewals_before_new_admissions(self):
        self.servers[1].vram_budget_mb = 220
        u, v = self.user(), self.user(1)
        self.start(u)
        u.prep.deadline_t = 5
        self.policy._trigger(v, 1, 0.5, 0.6)
        self.progress([u, v], t=0.5)
        self.assertIsNotNone(u.prep)
        self.assertIsNone(v.prep)
        self.assertEqual(self.memory.total(1), 155)

    def test_cancel_release_idempotent_and_new_session_owner(self):
        u = self.user()
        self.start(u)
        owner = self.memory.owner(u, u.prep)
        self.policy._cancel(u, self.params, self.metrics, 0.5)
        self.memory.sync([u])
        self.memory.release(owner)
        self.assertEqual(self.memory.stats['releases'], 1)
        u.session += 1
        self.start(u)
        self.assertNotEqual(owner, self.memory.owner(u, u.prep))

    def test_target_change_and_prediction_loss_release(self):
        u = self.user()
        self.start(u)
        self.policy.on_step(0.5, 0.5, [u], self.servers, self.params,
                            {0: Prediction(2, 1, 1)}, {}, self.metrics)
        self.progress([u], t=0.5)
        self.assertFalse(self.memory.entries)
        self.start(u)
        self.policy.on_step(0.5, 0.5, [u], self.servers, self.params, {}, {}, self.metrics)
        self.progress([u], t=0.5)
        self.assertFalse(self.memory.entries)

    def test_due_coordinated_plan_retained_on_deferral(self):
        p, u = Coordinated(detour=False), self.user()
        m = configure_memory(p, 'reserved', self.servers, self.params, self.metrics)
        self.servers[1].vram_budget_mb = 0
        p.on_step(0, 0.5, [u], self.servers, self.params, {0: Prediction(1, .2, 1)}, {}, self.metrics)
        planned = p.plans[0]['start']
        advance_preparations([u], self.servers, self.params, 0, .5, p.order_key, m)
        self.assertIn(0, p.plans)
        self.assertIsNone(u.prep)
        self.servers[1].vram_budget_mb = 200
        p.on_step(.1, .5, [u], self.servers, self.params, {0: Prediction(1, .2, 1)}, {}, self.metrics)
        advance_preparations([u], self.servers, self.params, .1, .5, p.order_key, m)
        self.assertNotIn(0, p.plans)
        self.assertEqual(m.events[-1]['planned_t'], planned)
        self.assertEqual(m.events[-1]['actual_t'], .1)

    def test_same_requests_same_approvals_across_policies(self):
        admitted = []
        for p in (PallasApprox(), Coordinated(detour=False)):
            m = configure_memory(p, 'reserved', self.servers, self.params, Metrics())
            users = [self.user(1), self.user(0)]
            for u in users:
                p._trigger(u, 1, 0, 1)
            advance_preparations(users, self.servers, self.params, 0, .5, p.order_key, m)
            admitted.append([u.id for u in users if u.prep is not None])
        self.assertEqual(admitted, [[0], [0]])

    def test_attempts_vs_contiguous_deferred_requests(self):
        u = self.user()
        self.servers[1].vram_budget_mb = 0
        for i in range(3):
            self.policy._trigger(u, 1, i * .5, 2)
            self.progress([u], t=i * .5)
        self.assertEqual(self.memory.stats['defer_attempts'], 3)
        self.assertEqual(self.memory.stats['deferred_requests'], 1)
        self.progress([u], t=1.5)  # request withdrawn
        self.policy._trigger(u, 1, 2, 3)
        self.progress([u], t=2)
        self.assertEqual(self.memory.stats['deferred_requests'], 2)

    def test_hedge_copy_release_and_winner_promotion(self):
        p, u = HedgedPallas(), self.user()
        m = configure_memory(p, 'reserved', self.servers, self.params, self.metrics)
        for target in (1, 2):
            p._trigger_copy(u, target, 0, 1, self.metrics)
        advance_preparations([u], self.servers, self.params, 0, .5, p.order_key, m)
        owners = list(m.entries)
        self.assertEqual(len(owners), 2)
        self.assertNotEqual(owners[0][2], owners[1][2])
        p.prepare_handover(u, self.servers[1], self.params, self.metrics, .5)
        m.sync([u])
        self.assertEqual(m.total(2), 0)
        self.assertEqual(m.total(1), 115)
        self.assertEqual(self.metrics.wasted_mb, 105)
        u.prep = None  # consumed by handover; not wasted
        m.sync([u])
        self.assertFalse(m.entries)

    def test_horizon_snapshot_then_cleanup_not_success_or_waste(self):
        u = self.user()
        self.start(u)
        self.memory.finish([u])
        self.assertEqual(self.memory.stats['end_pending'], 1)
        self.assertEqual(self.memory.stats['end_reserved_mb'], 115)
        self.assertEqual(self.memory.stats['end_actual_mb'], 105)
        self.assertFalse(self.memory.entries)
        self.assertEqual(self.metrics.prep_used, 0)
        self.assertEqual(self.metrics.wasted_mb, 0)

    def test_creation_failure_rolls_back_reservation(self):
        u = self.user()
        self.policy._trigger(u, 1, 0, 1)
        with patch('memory.Prep', side_effect=RuntimeError('creation failed')):
            with self.assertRaises(RuntimeError):
                self.progress([u])
        self.assertFalse(self.memory.entries)
        self.assertIsNone(u.prep)

    def test_unreserved_or_orphan_state_fails_loudly(self):
        u = self.user()
        u.prep = Prep(1, 1, 0, 1, 100, 100)
        with self.assertRaises(AssertionError):
            self.memory.check([u])

    def test_invalid_budget_and_unsupported_turn_boundary(self):
        for budget in (-1, float('nan'), float('inf')):
            self.servers[1].vram_budget_mb = budget
            with self.assertRaises(ValueError):
                configure_memory(self.policy, 'reserved', self.servers, self.params, self.metrics)
        with self.assertRaisesRegex(ValueError, 'BoundaryJob'):
            configure_memory(TurnBoundary(), 'reserved', self.servers, self.params, self.metrics)

    def test_settle_is_reserved_and_visible_to_gpu_planning(self):
        p = Coordinated()
        m = configure_memory(p, 'reserved', self.servers, self.params, self.metrics)
        u, v = self.user(), self.user(1, 1)
        u.server, u.hops, u.history = 1, 1, [(1, -20)]
        self.servers[1].vram_budget_mb = 500
        self.servers[1].prefill_speed = 10
        p.on_step(0, .5, [u], self.servers, self.params, {}, {}, self.metrics)
        advance_preparations([u], self.servers, self.params, 0, .5, p.order_key, m)
        self.assertIsNotNone(u.prep)
        self.assertEqual(m.total(1), 125)
        with patch.object(p, '_plan_target', wraps=p._plan_target) as plan:
            p.on_step(.5, .5, [u, v], self.servers, self.params,
                      {1: Prediction(1, 3, 1)}, {}, self.metrics)
            self.assertIn(u.prep, plan.call_args.args[2])

    def test_driver_respawn_and_return_release_without_leak(self):
        cfg = finalize_cfg(build_parser().parse_args(['--memory-mode', 'reserved', '--users', '1']))
        cfg.seed = 0
        servers = [Server(s, s * 100, 0, 500, 1000, 1000, 1000) for s in (0, 1, 2)]

        def snap(x, respawn=False):
            return [(x, 0, 0, 0, 100, respawn, 0, 0, 0, 0, False, 0, 0)]

        # Prepare -> respawn at same source -> distinct new prep -> migrate ->
        # prepare back to original source -> respawn with unfinished work.
        trace = [(snap(0), {0: Prediction(1, .2, 1)}),
                 (snap(0, True), {0: Prediction(1, .7, 1)}),
                 (snap(100), {0: Prediction(0, 1.2, 1)}),
                 (snap(0, True), {})]
        p = PallasApprox()
        result = run_policy(p, cfg, servers, trace)
        admissions = [e for e in p.memory.events if e['event'] == 'admit']
        self.assertEqual(len(admissions), 3)
        self.assertNotEqual(admissions[0]['owner'][1], admissions[1]['owner'][1])
        self.assertEqual(result['memory_releases'], 3)
        self.assertEqual(result['cancels'], 2)
        self.assertFalse(p.memory.entries)

        # Unpredicted handoff detours; prepare at third target; return to anchor
        # must cancel that preparation instead of consuming it.
        p = Coordinated()
        trace = [(snap(0), {}), (snap(100), {0: Prediction(2, .7, 1)}),
                 (snap(0), {})]
        result = run_policy(p, cfg, servers, trace)
        self.assertEqual(result['detours'], 1)
        self.assertEqual(result['memory_admissions'], 1)
        self.assertEqual(result['memory_releases'], 1)
        self.assertEqual(result['cancels'], 1)

    def test_driver_settle_completion_releases_reservation(self):
        cfg = finalize_cfg(build_parser().parse_args(['--memory-mode', 'reserved', '--users', '1']))
        cfg.seed = 0
        servers = [Server(s, s * 100, 0, 500, 1000, 1000, 1000) for s in (0, 1)]
        def snap(x):
            return [(x, 0, 0, 0, 100, False, 0, 0, 0, 0, False, 0, 0)]
        trace = [(snap(0), {})] + [(snap(100), {})] * 23
        p = Coordinated()
        run_policy(p, cfg, servers, trace)
        self.assertEqual(p.memory.metrics.settles, 1)
        self.assertEqual(p.memory.stats['admissions'], 1)
        self.assertEqual(p.memory.stats['releases'], 1)
        self.assertEqual(p.memory.stats['end_pending'], 0)

    def test_reserved_core_label_detour_off_legacy_unchanged(self):
        for mode, hops in [('legacy', 2), ('reserved', 0)]:
            cfg = finalize_cfg(build_parser().parse_args(['--memory-mode', mode]))
            p = policies_for(cfg)[-1]
            self.assertEqual(p.detour_hops, hops)
            self.assertEqual(p.name, 'coordinated-core-v1' if mode == 'reserved' else 'coordinated')

    def test_minisim_cleans_handover_reservations_and_mode_reuse(self):
        p = PallasApprox()
        result = mini_sim(p, 'qwen32b', 300, [1000, 1000], [5, 5], .5,
                          vram_mb=300, memory_mode='reserved')
        self.assertEqual(len(result), 2)
        self.assertGreater(p.memory.stats['admissions'], 0)
        self.assertFalse(p.memory.entries)
        self.assertEqual(p.memory.stats['end_pending'], 0)
        mini_sim(p, 'qwen32b', 300, [1000], [5], .5)
        self.assertIsNone(p.memory)

    def test_trace_and_closed_loop_driver_invariants(self):
        for clock in ('trace', 'closed-loop'):
            cfg = finalize_cfg(build_parser().parse_args([
                '--memory-mode', 'reserved', '--users', '12', '--steps', '160',
                '--vram-mb', '600', '--conversation-clock', clock]))
            cfg.seed = 0
            servers = build_servers(cfg, random.Random(999))
            trace = precompute_trace(cfg, servers)
            for p in (PallasApprox(), Coordinated(detour=False), HedgedPallas()):
                result = run_policy(p, cfg, servers, trace)
                self.assertEqual(result['memory_violations'], 0)
                self.assertLessEqual(result['memory_peak_reserved_target_mb'], 600)
                self.assertGreater(result['memory_admissions'], 0)
                self.assertFalse(p.memory.entries)


if __name__ == '__main__':
    unittest.main()
