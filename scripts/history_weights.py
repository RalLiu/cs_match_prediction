"""按观测场数统计历史；日期只用于排序和识别同日比赛批次。"""
from collections import Counter
import math

import numpy as np

MAP_HALF_LIFE_MATCHES = 20
RECENT_MAPS = 30
GLOBAL_RECENT_MAPS = 500


def batch_decay(count, half_life=MAP_HALF_LIFE_MATCHES):
    """同日 k 场先统一预测再更新；对未知场内顺序取各位置权重的平均。"""
    if count < 1 or not math.isfinite(half_life) or half_life <= 0:
        raise ValueError('批次场数和场数半衰期必须为正')
    step = 2 ** (-1 / half_life)
    decay = step ** count
    incoming = (1 - decay) / (count * (1 - step)) if step != 1 else 1.
    return decay, incoming


def series_weights(records):
    """每组同日对阵总权重相同，不使用距离当前日期的天数。"""
    counts = Counter((r['date'], r['a'], r['b']) for r in records)
    weights = np.array([1 / counts[(r['date'], r['a'], r['b'])] for r in records])
    return weights / weights.mean()
