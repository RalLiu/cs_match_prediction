"""两榜特征与仅使用历史赛果拟合的积分难度模型。"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from history_weights import series_weights

TIER_WIDTH = 80.0
TIER_SLOPE = 0.06
TIER_DIV = 5.0
RATING_PROVIDERS = ("hltv", "valve")


def build_tiers(rankings, medians, width=TIER_WIDTH, slope=TIER_SLOPE):
    """两榜各自以本榜最高分独立分档；缺榜使用本榜中位数对应档位。"""
    if not math.isfinite(width) or width <= 0 or not math.isfinite(slope) or slope < 0:
        raise ValueError("档宽须为有限正数，档宽斜率须为有限非负数")
    tiers, fallback = {}, {}
    for provider in RATING_PROVIDERS:
        hi = max(rankings[provider].values())

        def index(points):
            steps, x = 0, float(points)
            while x < hi:
                x += width + slope * x
                steps += 1
            return steps

        tiers[provider] = {t: index(p) for t, p in rankings[provider].items()}
        fallback[provider] = index(medians[provider])
    return tiers, fallback


def tier_features(tiers, fallback):
    def ratings(r):
        return tuple((tiers[p].get(r["a"], fallback[p])
                      - tiers[p].get(r["b"], fallback[p])) / TIER_DIV
                     for p in RATING_PROVIDERS)
    return ratings


def rating_pair(rankings, medians, engine="tier", width=TIER_WIDTH, slope=TIER_SLOPE):
    if engine == "tier":
        return tier_features(*build_tiers(rankings, medians, width, slope))
    if engine != "points":
        raise ValueError("engine 必须为 tier 或 points")

    def ratings(r):
        return tuple(math.log1p(rankings[p].get(r["a"], medians[p]))
                     - math.log1p(rankings[p].get(r["b"], medians[p]))
                     for p in RATING_PROVIDERS)
    return ratings


class RankingDifficulty:
    """按日期先 fit 过去、再 probability、最后 update 当天；不使用地图表现。"""
    def __init__(self, rankings, medians, engine="tier", width=TIER_WIDTH, slope=TIER_SLOPE):
        self.ratings = rating_pair(rankings, medians, engine, width, slope)
        self.bounds = [(None, 0)] * 2 if engine == "tier" else [(0, None)] * 2
        self.records = []
        self.weights = np.zeros(2)

    def fit(self, as_of):
        if not self.records:
            return
        if any(r["date"] >= as_of for r in self.records):
            raise ValueError("积分难度拟合只能使用目标日期之前的赛果")
        x = np.array([self.ratings(r) for r in self.records])
        scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
        xn = x / scale
        y = np.array([r["y"] for r in self.records])
        sample_weights = series_weights(self.records)

        def objective(theta):
            z = xn @ theta
            loss = np.sum(sample_weights * (np.logaddexp(0, z) - y * z)) + .01 * theta @ theta
            gradient = xn.T @ (sample_weights * (expit(z) - y)) + .02 * theta
            return float(loss), gradient

        fitted = minimize(objective, self.weights * scale, jac=True,
                          method="L-BFGS-B", bounds=self.bounds)
        self.weights = fitted.x / scale

    def probability(self, a, b):
        return float(expit(np.array(self.ratings(dict(a=a, b=b))) @ self.weights))

    def update(self, records):
        self.records.extend(records)
