"""日历间隔不影响特征、拟合与 BP；场数衰减检查。"""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import build_training_features, predict_features  # noqa: E402
from history_weights import batch_decay, series_weights  # noqa: E402
from map_preference_features import preference  # noqa: E402
from series_probability import build_profiles, fit_strength, build_tiers, tier_features  # noqa: E402


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.rankings = {'hltv': {'a': 100, 'b': 1000}, 'valve': {'a': 1000, 'b': 2000}}
        self.medians = {'hltv': 550, 'valve': 1500}
        self.rows = [dict(date=f'2026-11-{i:02}', a='a', b='b', map='Ancient' if i%2 else 'Nuke',
                          sa=13 if i%3 else 7, sb=7 if i%3 else 13, y=float(i%3 != 0))
                     for i in range(1, 7)]
        self.spread = copy.deepcopy(self.rows)
        for i, row in enumerate(self.spread):
            row['date'] = f'{2020+i}-01-01'

    def test_gap_spacing_changes_neither_features_nor_fitted_models(self):
        for engine in ('tier', 'points'):
            original, a = build_training_features(self.rows, self.rankings, self.medians, engine=engine)
            spread, b = build_training_features(self.spread, self.rankings, self.medians, engine=engine)
            np.testing.assert_allclose([r['x4'] for r in original], [r['x4'] for r in spread], atol=1e-12)
            ratings = tier_features(*build_tiers(self.rankings, self.medians)) if engine=='tier' else None
            fitted_a, fitted_b = fit_strength(original, ratings), fit_strength(spread, ratings)
            np.testing.assert_allclose(fitted_a['weights'], fitted_b['weights'], atol=1e-12)
            self.assertAlmostEqual(fitted_a['c'], fitted_b['c'])
            pool = ['Ancient', 'Nuke']
            profiles_a = build_profiles('a', 'b', '2030-01-01', a, original, pool)
            profiles_b = build_profiles('a', 'b', '2030-01-01', b, spread, pool)
            self.assertEqual(profiles_a, profiles_b)

    def test_prediction_date_does_not_change_features_or_bp_without_new_matches(self):
        _, model = build_training_features(self.rows, self.rankings, self.medians)
        for as_of in ('2026-11-07', '2030-01-01'):
            features = predict_features(model, 'a', 'b', 'Ancient', as_of)
            profiles = build_profiles('a', 'b', as_of, model, self.rows, ['Ancient','Nuke'])
            if as_of=='2026-11-07':
                first_features, first_profiles = features, profiles
            else:
                self.assertEqual(features, first_features)
                self.assertEqual(profiles, first_profiles)

    def test_count_half_life_and_same_day_batch_average(self):
        decay, incoming = batch_decay(20)
        self.assertAlmostEqual(decay, .5)
        q = 2**(-1/20)
        self.assertAlmostEqual(20*incoming, sum(q**i for i in range(20)))
        self.assertAlmostEqual(batch_decay(10)[0], 2**(-.5))
        with self.assertRaises(ValueError):
            batch_decay(0)

    def test_series_weight_is_independent_of_calendar_gaps(self):
        np.testing.assert_array_equal(series_weights(self.rows), series_weights(self.spread))
        np.testing.assert_array_equal(series_weights(self.rows), np.ones(6))

    def test_bp_recent_is_last_thirty_maps_not_last_thirty_days(self):
        history = [(f'{1900+i}-01-01', 'Ancient' if i<10 else 'Nuke') for i in range(40)]
        p = preference(history, 'Ancient', '2030-01-01')
        self.assertEqual(p['recent_total'], 30)
        self.assertEqual(p['recent_count'], 0)
        self.assertEqual(p['previous_count'], 10)

    def test_playing_other_maps_decays_every_old_map_once_per_match(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['map'] = 'Ancient'
        for row in rows[1:]:
            row['map'] = 'Nuke'
        _, before = build_training_features(rows[:1], self.rankings, self.medians)
        _, after = build_training_features(rows, self.rankings, self.medians)
        q = 2**(-1/20)
        for team in ('a','b'):
            old = before['state'][team]['maps']['Ancient']
            new = after['state'][team]['maps']['Ancient']
            self.assertAlmostEqual(new['n'], old['n']*q**5)
            self.assertAlmostEqual(new['margin'], old['margin']*q**5)
            # 无新 Ancient 数据时只改变质量，不凭空改变未收缩平均数。
            self.assertAlmostEqual(new['margin']/new['n'], old['margin']/old['n'])

    def test_replayed_map_gives_new_observation_more_weight_after_other_maps(self):
        rows = copy.deepcopy(self.rows)
        for row in rows[1:-1]:
            row['map'] = 'Nuke'
        rows[0]['map'] = rows[-1]['map'] = 'Ancient'
        _, model = build_training_features(rows, self.rankings, self.medians)
        q = 2**(-1/20)
        self.assertAlmostEqual(model['state']['a']['maps']['Ancient']['n'], q**5+1)

    def test_same_day_all_maps_batch_uses_team_count_not_per_map_count(self):
        rows = copy.deepcopy(self.rows[:3])
        rows[0]['map'] = 'Ancient'
        rows[1]['map'] = 'Nuke'
        rows[2]['map'] = 'Ancient'
        rows[2]['date'] = rows[1]['date']
        _, model = build_training_features(rows, self.rankings, self.medians)
        q = 2**(-1/20)
        expected = q**2+(1+q)/2
        self.assertAlmostEqual(model['state']['a']['maps']['Ancient']['n'], expected)
        swapped = [rows[0],rows[2],rows[1]]
        _, other = build_training_features(swapped, self.rankings, self.medians)
        np.testing.assert_allclose(model['state']['a']['maps']['Ancient']['margin'],
                                   other['state']['a']['maps']['Ancient']['margin'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
