"""Focused regression tests for the three alternative baselines."""

import random
import unittest

from environment import CostParams, LinkLoad, Prep, Server, User, advance_preparations
from metrics import Metrics
from policies import HedgedPallas, TurnBoundary
from prediction import predict_candidates
from run import build_parser, finalize_cfg


class AlternativeBaselineTests(unittest.TestCase):
    def test_ideal_compression_scales_kv_bytes(self):
        cfg = finalize_cfg(build_parser().parse_args(["--kv-compression-ratio", "0.25"]))
        self.assertAlmostEqual(cfg.raw_kv_mb_per_token, 0.25)
        self.assertAlmostEqual(cfg.kv_mb_per_token, 0.0625)

    def test_hedged_copies_consume_each_target_resource(self):
        servers = {
            sid: Server(sid, float(sid), 0.0, 1.0, 100.0, 100.0, 1000.0, 1.0)
            for sid in (1, 2)
        }
        user = User(0, 0.0, 0.0)
        user.hedge_preps = {
            sid: Prep(sid, sid, 0.0, 2.0, 100.0, 100.0) for sid in (1, 2)
        }
        loads = advance_preparations([user], servers, CostParams(), 0.0, 0.5)
        self.assertAlmostEqual(user.hedge_preps[1].prefix_remaining, 50.0)
        self.assertAlmostEqual(user.hedge_preps[2].prefix_remaining, 50.0)
        self.assertEqual(loads[1].prefills, 1)
        self.assertEqual(loads[2].prefills, 1)

    def test_hedge_promotes_hit_and_cancels_loser(self):
        user = User(0, 0.0, 0.0)
        user.hedge_preps = {
            sid: Prep(sid, sid, 0.0, 2.0, 100.0, 50.0) for sid in (1, 2)
        }
        dst = Server(2, 0.0, 0.0, 1.0, 100.0, 100.0, 1000.0)
        metrics = Metrics()
        policy = HedgedPallas()
        policy.primary_targets[user.id] = 1
        policy.prepare_handover(user, dst, CostParams(), metrics, 1.0)
        self.assertEqual(user.prep.target, 2)
        self.assertFalse(user.hedge_preps)
        self.assertEqual(metrics.hedge_hits, 1)
        self.assertEqual(metrics.hedge_alt_hits, 1)
        self.assertEqual(metrics.cancels, 1)

    def test_turn_boundary_hides_migration_inside_think_time(self):
        servers = {1: Server(1, 0.0, 0.0, 1.0, 1000.0, 1000.0, 1000.0)}
        user = User(0, 0.0, 0.0, tokens=100.0, server=1, anchor=0, hops=1,
                    turn_remaining_s=0.0, think_remaining_s=5.0)
        metrics = Metrics()
        TurnBoundary().on_step(0.0, 0.5, [user], servers, CostParams(), {},
                               {1: LinkLoad()}, metrics)
        self.assertEqual(user.anchor, 1)
        self.assertEqual(user.hops, 0)
        self.assertEqual(metrics.boundary_moves, 1)
        self.assertEqual(metrics.boundary_hidden, 1)
        self.assertAlmostEqual(metrics.sits[0], 0.0)

    def test_turn_boundary_job_occupies_resources_across_steps(self):
        servers = {1: Server(1, 0.0, 0.0, 1.0, 10.0, 1.0, 1000.0, 1.0)}
        users = [User(i, 0.0, 0.0, tokens=40.0, server=1, anchor=0, hops=1,
                      think_remaining_s=0.5) for i in range(2)]
        policy, metrics = TurnBoundary(), Metrics()
        first = {1: LinkLoad()}
        policy.on_step(0.0, 0.5, users, servers, CostParams(), {}, first, metrics)
        self.assertEqual([u.anchor for u in users], [0, 0])
        self.assertEqual(first[1].streams, 2)
        self.assertEqual(first[1].prefills, 2)
        self.assertGreater(first[1].sent_mb, 0.0)
        self.assertLessEqual(first[1].sent_mb, 0.5 + 1e-9)
        self.assertEqual(metrics.boundary_moves, 0)
        policy.finalize(metrics)
        self.assertEqual(metrics.boundary_pending, 2)

        for i in range(1, 30):
            policy.on_step(i * 0.5, 0.5, users, servers, CostParams(), {},
                           {1: LinkLoad()}, metrics)
            if metrics.boundary_moves == 2:
                break
        self.assertEqual(metrics.boundary_moves, 2)
        self.assertEqual([u.anchor for u in users], [1, 1])
        self.assertTrue(all(s > 0.0 for s in metrics.sits))
        policy.finalize(metrics)
        self.assertEqual(metrics.boundary_pending, 0)

    def test_turn_boundary_waits_for_actual_boundary_offset(self):
        servers = {1: Server(1, 0.0, 0.0, 1.0, 1000.0, 1000.0, 1000.0)}
        user = User(0, 0.0, 0.0, tokens=100.0, server=1, anchor=0, hops=1,
                    boundary_crossed_step=True, boundary_offset_s=0.48,
                    boundary_gap_s=1.0, think_remaining_s=0.98)
        policy, metrics = TurnBoundary(), Metrics()
        policy.on_step(0.0, 0.5, [user], servers, CostParams(), {},
                       {1: LinkLoad()}, metrics)
        self.assertEqual(user.anchor, 0)
        self.assertEqual(metrics.boundary_moves, 0)
        policy.on_step(0.5, 0.5, [user], servers, CostParams(), {},
                       {1: LinkLoad()}, metrics)
        self.assertEqual(user.anchor, 1)
        self.assertAlmostEqual(metrics.sits[0], 0.0)

    def test_turn_boundary_cancels_stale_target(self):
        servers = {i: Server(i, float(i), 0.0, 1.0, 10.0, 1.0, 1000.0)
                   for i in (1, 2)}
        user = User(0, 0.0, 0.0, tokens=100.0, server=1, anchor=0, hops=1,
                    think_remaining_s=1.0)
        policy, metrics = TurnBoundary(), Metrics()
        policy.on_step(0.0, 0.5, [user], servers, CostParams(), {},
                       {i: LinkLoad() for i in servers}, metrics)
        self.assertIn(user.id, policy.jobs)
        user.server = 2
        policy.on_step(0.5, 0.5, [user], servers, CostParams(), {},
                       {i: LinkLoad() for i in servers}, metrics)
        self.assertEqual(metrics.boundary_cancels, 1)
        self.assertGreater(metrics.wasted_mb, 0.0)
        self.assertEqual(user.anchor, 0)
        self.assertEqual(policy.jobs[user.id].target, 2)

    def test_candidate_probabilities_keep_missing_mass(self):
        servers = [
            Server(0, 0.0, 0.0, 100.0, 100.0, 100.0, 1000.0),
            Server(1, 200.0, 0.0, 100.0, 100.0, 100.0, 1000.0),
            Server(2, 0.0, 200.0, 100.0, 100.0, 100.0, 1000.0),
        ]
        user = User(0, 20.0, 20.0, vx=20.0, vy=0.0, server=0)
        pred = predict_candidates(user, servers, 0.0, 20.0, 0.5, random.Random(4),
                                  0.2, 0.5, candidate_count=2, samples=32)
        self.assertIsNotNone(pred)
        self.assertLessEqual(sum(c.probability for c in pred.candidates), 1.0)
        self.assertEqual(pred.target, pred.candidates[0].target)
        self.assertGreaterEqual(pred.candidates[0].probability,
                                pred.candidates[-1].probability)


if __name__ == "__main__":
    unittest.main()
