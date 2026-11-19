"""双积分引擎检查：python scripts/test/test_strength.py。"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import predict_features  # noqa: E402
from series_probability import build_tiers, fit_strength, strength_row, tier_features  # noqa: E402


class StrengthTests(unittest.TestCase):
    def setUp(self):
        # 两个榜单反向排序，保证测试能识别只变更其中一榜或误用同一榜。
        self.rankings = {"hltv": {"a": 1000, "b": 900, "c": 0},
                         "valve": {"a": 0, "b": 900, "c": 1000}}
        self.medians = {"hltv": 900, "valve": 900}

    def test_both_tiers_are_independent_and_reversible(self):
        tiers, fallback = build_tiers(self.rankings, self.medians, width=100, slope=0)
        self.assertEqual(tiers["hltv"], {"a": 0, "b": 1, "c": 10})
        self.assertEqual(tiers["valve"], {"a": 10, "b": 1, "c": 0})
        ratings = tier_features(tiers, fallback)
        np.testing.assert_allclose(ratings(dict(a="a", b="b")), [-.2, 1.8])
        np.testing.assert_allclose(ratings(dict(a="b", b="a")), [.2, -1.8])
        self.assertEqual(ratings(dict(a="missing", b="b")), (0., 0.))

    def test_same_tier_discards_within_tier_scores_in_both_columns(self):
        rankings = {"hltv": {"a": 1000, "b": 950, "c": 940},
                    "valve": {"a": 2000, "b": 1950, "c": 1940}}
        tiers, fallback = build_tiers(rankings, self.medians, width=100, slope=0)
        self.assertEqual(tier_features(tiers, fallback)(dict(a="b", b="c")), (0., 0.))

    def test_points_uses_log1p_for_both_rankings(self):
        model = dict(rankings=self.rankings, medians=self.medians, elo={}, state={})
        x, _, _, _ = predict_features(model, "a", "b", "Ancient", "2026-11-19")
        expected = [math.log1p(1000) - math.log1p(900), -math.log1p(900), 0.]
        np.testing.assert_allclose(strength_row(x), expected)
        reverse = predict_features(model, "b", "a", "Ancient", "2026-11-19")[0]
        np.testing.assert_allclose(strength_row(reverse), -strength_row(x))
        missing = predict_features(model, "missing", "b", "Ancient", "2026-11-19")[0]
        np.testing.assert_allclose(strength_row(missing), [0., 0., 0.])

    def test_training_uses_the_same_dual_tier_features_as_prediction(self):
        tiers, fallback = build_tiers(self.rankings, self.medians, width=100, slope=0)
        ratings = tier_features(tiers, fallback)
        rows = []
        for i, (a, b) in enumerate([("a", "b"), ("b", "c"), ("c", "a")] * 8):
            rows.append(dict(a=a, b=b, date="2026-11-17", y=float(i % 3 != 0),
                             avoid=.1 * (i % 2), x4=[9., 8., 0., 0., 0., 0., .01 * i]))
        transformed = copy.deepcopy(rows)
        for row in transformed:
            row["x4"][:2] = ratings(row)
        fitted = fit_strength(rows, ratings)
        explicit = fit_strength(transformed)
        np.testing.assert_allclose(fitted["weights"], explicit["weights"], rtol=1e-12)
        self.assertAlmostEqual(fitted["c"], explicit["c"], places=12)
        for row in rows:
            np.testing.assert_allclose(strength_row(row["x4"], ratings(row))[:2], ratings(row))

    def test_invalid_tier_parameters_are_rejected(self):
        for width, slope in [(0, 0), (-1, 0), (100, -1), (float("nan"), 0), (100, float("inf"))]:
            with self.subTest(width=width, slope=slope), self.assertRaises(ValueError):
                build_tiers(self.rankings, self.medians, width=width, slope=slope)


if __name__ == "__main__":
    unittest.main(verbosity=2)
