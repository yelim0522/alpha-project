"""Causal-clock invariants, including right censoring and resource conservation."""

import random
import unittest

from conversation import Conversation, ConversationClock, Workload
from environment import CostParams, User
from metrics import Metrics
from policies import Detour, TurnBoundary
from run import build_parser, finalize_cfg, build_servers, precompute_trace, run_policy


class ConversationTests(unittest.TestCase):
    def setup_clock(self, phase="generate", remaining=10.0, think=0.0):
        cfg = finalize_cfg(build_parser().parse_args([
            "--conversation-clock", "closed-loop", "--decode-rate", "10",
            "--turn-output-tokens", "10", "--think-time", "1"]))
        cfg.seed = 3
        metrics = Metrics()
        clock = ConversationClock(cfg, metrics)
        u = User(0, 0, 0, tokens=100, server=1, anchor=1)
        clock.states[0] = Conversation(random.Random(1), random.Random(2), remaining, 1,
                                      phase, think, request_t=0.0, response_tokens=10)
        return cfg, clock, u, metrics

    def test_recovery_pauses_tokens_and_delays_response(self):
        cfg, clock, u, _ = self.setup_clock()
        u.recovery_until_s = 1.5
        clock.advance([u], Detour(), 0, 2, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.tokens, 105)
        self.assertAlmostEqual(clock.summary()["observed_wait_s"], 1.5)
        clock.advance([u], Detour(), 2, 0.5, CostParams(decode_rate=10))
        result = clock.summary()
        self.assertEqual(result["responses_completed"], 1)
        self.assertAlmostEqual(result["response_e2e_mean_s"], 2.5)
        self.assertAlmostEqual(result["response_excess_mean_s"], 1.5)

    def test_thinking_overlaps_recovery_without_user_wait(self):
        _, clock, u, _ = self.setup_clock(phase="think", remaining=0, think=2)
        u.recovery_until_s = 1.0
        clock.advance([u], Detour(), 0, 2.5, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.tokens, 105)
        self.assertAlmostEqual(clock.summary()["observed_wait_s"], 0)

    def test_boundary_gate_preserves_next_response(self):
        _, clock, u, _ = self.setup_clock(phase="ready")
        u.anchor, u.hops = 0, 1
        clock.advance([u], TurnBoundary(), 0, 0.5, CostParams(decode_rate=10))
        self.assertEqual(u.tokens, 100)
        self.assertEqual(clock.states[0].phase, "ready")
        # Migration finishes 0.3 s into the following step.
        u.anchor, u.hops, u.boundary_ready_t = 1, 0, 0.8
        clock.advance([u], TurnBoundary(), 0.5, 0.5, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.tokens, 102)
        self.assertAlmostEqual(clock.summary()["observed_wait_s"], 0.8)

    def test_unfinished_generation_can_finish_but_next_turn_waits(self):
        _, clock, u, _ = self.setup_clock(remaining=1)
        clock.states[0].gap = 0
        u.anchor, u.hops = 0, 1
        clock.advance([u], TurnBoundary(), 0, 0.5, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.tokens, 101)
        self.assertEqual(clock.states[0].phase, "ready")
        self.assertGreater(clock.summary()["observed_wait_s"], 0)

    def test_censored_wait_is_not_discarded_or_double_counted(self):
        _, clock, u, _ = self.setup_clock(phase="ready")
        u.anchor = 0
        clock.advance([u], TurnBoundary(), 0, 1, CostParams(decode_rate=10))
        result = clock.summary()
        self.assertEqual(result["censored_wait_s"], 1)
        self.assertEqual(result["censored_wait_episodes"], 1)
        self.assertEqual(result["responses_completed"], 0)
        self.assertEqual(result, clock.summary())
        clock.reset_user(u)
        self.assertEqual(clock.summary()["censored_wait_s"], 1)

    def test_decode_budget_and_forwarding_slow_generation(self):
        cfg, clock, u, _ = self.setup_clock()
        cfg.decode_capacity = 5
        clock.advance([u], Detour(), 0, 1, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.tokens, 105)
        self.assertAlmostEqual(clock.summary()["generated_tokens"], 5)
        cfg.decode_capacity = 0
        u.anchor, u.hops = 0, 1
        clock.advance([u], Detour(), 1, 0.5, CostParams(decode_rate=10, itl_hop_penalty_ms=100))
        self.assertAlmostEqual(u.generated_tokens_step, 2.5)

    def test_workload_is_reproducible_and_variable(self):
        cfg, _, _, _ = self.setup_clock()
        cfg.turn_distribution, cfg.think_distribution = "lognormal", "mixture"
        workload = Workload(cfg)
        def sample():
            a, b = random.Random(7), random.Random(9)
            return [workload.sample(a, b) for _ in range(100)]
        values = sample()
        self.assertEqual(values, sample())
        self.assertGreater(len(set(length for length, _ in values)), 1)
        self.assertTrue(any(gap == 0 for _, gap in values))

    def test_shared_decode_capacity_and_remote_memory_accounting(self):
        cfg, clock, u, _ = self.setup_clock()
        cfg.decode_capacity = 10
        v = User(1, 0, 0, tokens=200, anchor=1, server=2, hops=1)
        clock.states[1] = Conversation(random.Random(3), random.Random(4), 10, 1,
                                      request_t=0, response_tokens=10)
        clock.account_residence([u, v], 1, CostParams(decode_rate=10))
        clock.advance([u, v], Detour(), 0, 1, CostParams(decode_rate=10))
        self.assertAlmostEqual(u.generated_tokens_step + v.generated_tokens_step, 10)
        self.assertAlmostEqual(clock.summary()["retained_kv_gb_s"], 0.05)
        self.assertAlmostEqual(clock.summary()["peak_resident_kv_mb"], 75)

    def test_replay_is_deterministic_and_accounts_every_wait(self):
        cfg = finalize_cfg(build_parser().parse_args([
            "--conversation-clock", "closed-loop", "--users", "8", "--steps", "100",
            "--turn-output-tokens", "16", "--think-time", "0"]))
        cfg.seed = 7
        servers = build_servers(cfg, random.Random(1006))
        trace = precompute_trace(cfg, servers)
        a = run_policy(TurnBoundary(), cfg, servers, trace)
        b = run_policy(TurnBoundary(), cfg, servers, trace)
        self.assertEqual(a, b)
        self.assertGreater(a["handovers"], 0)
        self.assertGreater(a["generated_tokens"], 0)
        self.assertAlmostEqual(a["observed_wait_s"], a["completed_wait_s"] + a["censored_wait_s"])


if __name__ == "__main__":
    unittest.main()
