"""Audit plotting statistics without needing matplotlib."""

import copy
import json
from pathlib import Path
import unittest

from plot_turn_validation import paired_effects, read_rows, validate_rows, validate_summary


class PlotAuditTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.rows = read_rows(root / "turn_validation_v09_seeds.csv")
        self.summaries = read_rows(root / "turn_validation_v09_summary.csv")
        self.seeds = json.loads((root / "turn_validation_v09_manifest.json").read_text())["seeds"]

    def test_committed_data_and_summary_agree(self):
        data = validate_rows(self.rows, self.seeds)
        self.assertEqual(len(data), 84)
        validate_summary(data, self.summaries, self.seeds)

    def test_incomplete_and_duplicate_design_fail(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError):
                validate_rows(rows, self.seeds)

    def test_nonfinite_metric_fails(self):
        self.rows[0]["response_wait_p99_s"] = "nan"
        with self.assertRaises(ValueError):
            validate_rows(self.rows, self.seeds)

    def test_changed_summary_fails(self):
        data = validate_rows(self.rows, self.seeds)
        changed = copy.deepcopy(self.summaries)
        changed[0]["response_wait_p99_s_mean"] = "100"
        with self.assertRaises(ValueError):
            validate_summary(data, changed, self.seeds)

    def test_ratios_pair_by_seed_not_row_order(self):
        data = validate_rows(list(reversed(self.rows)), self.seeds)
        effects = paired_effects(data, self.seeds, "azure-mixture")
        self.assertAlmostEqual(effects["completion_ratio"][0], 9026 / 10160)
        self.assertTrue(all(v > 0 for v in effects["wait_p99_difference_s"]))


if __name__ == "__main__":
    unittest.main()
