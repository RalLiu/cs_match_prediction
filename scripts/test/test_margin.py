"""积分加权净胜、不收缩与时间顺序检查。python scripts/test/test_margin.py"""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_weights import batch_decay  # noqa: E402
from features import build_training_features, predict_features, summarize, weighted_margin  # noqa: E402
from rating_features import RankingDifficulty  # noqa: E402


class MarginTests(unittest.TestCase):
    def setUp(self):
        self.rankings = {'hltv': {'a': 100, 'b': 1000}, 'valve': {'a': 1000, 'b': 2000}}
        self.medians = {'hltv': 550, 'valve': 1500}
        self.observations = [dict(date=f'2026-11-{day:02d}', a='a', b='b', map='Ancient',
                                  sa=13 if day % 2 else 7, sb=7 if day % 2 else 13,
                                  y=float(day % 2)) for day in range(1, 6)]

    def test_opponent_difficulty_and_direction(self):
        self.assertAlmostEqual(weighted_margin(.25, .2), .4)
        self.assertAlmostEqual(weighted_margin(-.25, .2), -.1)
        self.assertAlmostEqual(weighted_margin(.25, .8), .1)
        self.assertAlmostEqual(weighted_margin(-.25, .8), -.4)
        for net in (-.8, -.1, 0., .2, .9):
            for p in (.01, .2, .5, .9):
                self.assertAlmostEqual(weighted_margin(net, p), -weighted_margin(-net, 1-p))

    def test_one_map_is_not_shrunk_and_empty_map_is_zero(self):
        _, model = build_training_features(self.observations[:1], self.rankings, self.medians)
        a = summarize(model['state']['a'], 'Ancient', '2026-11-02')
        b = summarize(model['state']['b'], 'Ancient', '2026-11-02')
        self.assertAlmostEqual(a['margin'], .3)  # 首日无历史，积分难度 p=.5，不是 .3/9。
        self.assertAlmostEqual(b['margin'], -.3)
        self.assertEqual(summarize(model['state']['a'], 'Nuke', '2026-11-02')['margin'], 0.)

    def test_match_half_life_and_no_decay_while_idle(self):
        state = dict(timeline=[], maps={'Ancient': dict(
                     n=1., margin=.4, delta=0., result_residual=0., margin_shrinkage=0)})
        self.assertAlmostEqual(summarize(state, 'Ancient', '2026-12-01')['margin'], .4)
        self.assertAlmostEqual(batch_decay(20)[0], .5)
        self.assertAlmostEqual(summarize(state, 'Ancient', '2030-12-01')['margin'], .4)
        # 半衰期按该队任意地图新增观测场数，不是经过的天数。
        decay, incoming = batch_decay(20)
        state['maps']['Ancient'].update(n=1.*decay+20*incoming,
                                        margin=.4*decay+(-.2)*20*incoming)
        expected = (.4*.5-.2*20*incoming)/(.5+20*incoming)
        self.assertAlmostEqual(summarize(state, 'Ancient', '2030-12-01')['margin'], expected)

    def test_future_results_do_not_change_earlier_features(self):
        for engine in ('tier', 'points'):
            full, _ = build_training_features(self.observations, self.rankings, self.medians, engine=engine)
            prefix, _ = build_training_features(self.observations[:3], self.rankings, self.medians, engine=engine)
            changed = copy.deepcopy(self.observations)
            for r in changed[3:]:
                r['sa'], r['sb'], r['y'] = r['sb'], r['sa'], 1-r['y']
            other, _ = build_training_features(changed, self.rankings, self.medians, engine=engine)
            for i in range(3):
                np.testing.assert_allclose(full[i]['x4'], prefix[i]['x4'])
                np.testing.assert_allclose(full[i]['x4'], other[i]['x4'])

    def test_same_day_games_share_pre_day_features(self):
        data = copy.deepcopy(self.observations)
        data[1]['date'] = data[0]['date']
        rows, _ = build_training_features(data, self.rankings, self.medians)
        np.testing.assert_allclose(rows[0]['x4'], rows[1]['x4'])
        self.assertEqual(rows[0]['x4'][6], 0.)

    def test_difficulty_has_no_map_feature_and_rejects_same_day_labels(self):
        for engine in ('tier', 'points'):
            difficulty = RankingDifficulty(self.rankings, self.medians, engine)
            self.assertEqual(difficulty.probability('a', 'b'), .5)
            difficulty.update([self.observations[1]])
            with self.assertRaises(ValueError):
                difficulty.fit('2026-11-02')
            difficulty.fit('2026-11-03')
            self.assertLess(difficulty.probability('a', 'b'), .5)
            self.assertAlmostEqual(difficulty.probability('a', 'b') + difficulty.probability('b', 'a'), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
