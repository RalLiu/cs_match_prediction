"""前向折 OOF 评估：地图级 + 系列赛级，tier / points 两个引擎对照。

用法（项目根目录）：
    python scripts/evaluate.py
不写任何文件，只打印对照表。
"""
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_io import load_maps, load_rankings, ordinal  # noqa: E402
from features import build_training_features  # noqa: E402
from series_probability import (RAIL, TIER_SLOPE, TIER_WIDTH, build_tiers,  # noqa: E402
                                fit_strength, strength_row, tier_second)

FOLDS = [.40, .55, .70, .85]


def ll_of(y, p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-y * np.log(q) - (1 - y) * np.log1p(-q)))


def series_cond(p1, p2, p3):
    """已知 3 图打满时，A 以 2-1 取胜的条件概率。"""
    win = p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
    lose = (1 - p1) * (1 - p2) * p3 + (1 - p1) * p2 * (1 - p3) + p1 * (1 - p2) * (1 - p3)
    return win / (win + lose)


def main():
    observations, names, manifest, raw, quarantine = load_maps()
    rankings, medians, _ = load_rankings()
    rows, current = build_training_features(observations, rankings, medians)
    dates = sorted({r["date"] for r in rows})
    cuts = [dates[int(len(dates) * q)] for q in FOLDS]
    for r in rows:
        r["level"] = (rankings["valve"].get(r["a"], medians["valve"])
                      + rankings["valve"].get(r["b"], medians["valve"])) / 2

    tiers, fallback = build_tiers(rankings, medians, TIER_WIDTH, TIER_SLOPE)
    print(f"数据: {len(rows)} 张地图, {dates[0]} .. {dates[-1]}, 折 = {cuts}")
    print(f"分档: 宽度 {TIER_WIDTH:.0f} + {TIER_SLOPE:.0%}×积分, "
          f"共 {max(tiers.values()) + 1} 档\n")

    table = {}
    for engine in ("tier", "points"):
        second = tier_second(tiers, fallback) if engine == "tier" else None
        pred, rec = [], []
        for start, end in zip(cuts[:-1], cuts[1:]):
            tr = [r for r in rows if r["date"] < start]
            fu = [r for r in rows if start <= r["date"] < end]
            st = fit_strength(tr, second)
            w, c = np.array(st["weights"]), st["c"]
            for r in fu:
                col2 = second(dict(a=r["a"], b=r["b"])) if second else r["x4"][1]
                z = float(strength_row(r["x4"], col2) @ w - c * r["avoid"])
                r["_p"] = float(min(max(1 / (1 + np.exp(-z)), RAIL), 1 - RAIL))
            pred += [r["_p"] for r in fu]
            rec += fu
        p = np.array(pred)
        y = np.array([r["y"] for r in rec])
        lv = np.array([r["level"] for r in rec])
        # 系列赛级：必须在切换到下一个引擎之前算完（行对象是共用的）
        groups = defaultdict(list)
        for r in rec:
            groups[(r["date"], r["a"], r["b"])].append(r)
        sys_y, sys_p = [], []
        for _, v in groups.items():
            if len(v) != 3:
                continue
            sys_p.append(series_cond(*[r["_p"] for r in v]))
            sys_y.append(1.0 if sum(r["y"] for r in v) >= 2 else 0.0)
        table[engine] = dict(
            ll=ll_of(y, p), brier=float(np.mean((p - y) ** 2)),
            acc=float(np.mean((p >= .5) == y)),
            ll16=ll_of(y[lv >= 1600], p[lv >= 1600]),
            ll17=ll_of(y[lv >= 1700], p[lv >= 1700]),
            series=(np.array(sys_y), np.array(sys_p)),
            rows=rec, p=p,
            same_idx=[i for i, r in enumerate(rec)
                      if tiers.get(r["a"], fallback) == tiers.get(r["b"], fallback)])

    print("=== 地图级（3 折合并 OOF）===")
    print(f"  {'引擎':8s}{'LogLoss':>10s}{'Brier':>9s}{'Acc':>9s}{'≥1600':>9s}{'≥1700':>9s}")
    for k, v in table.items():
        print(f"  {k:8s}{v['ll']:10.4f}{v['brier']:9.4f}{v['acc']:9.4f}{v['ll16']:9.4f}{v['ll17']:9.4f}")

    print("\n=== 系列赛级（仅 3 图打满、且三图都有 OOF 预测）===")
    print(f"  {'引擎':8s}{'n':>6s}{'LogLoss':>10s}{'Acc':>9s}")
    for engine, v in table.items():
        ys, ps = v["series"]
        print(f"  {engine:8s}{len(ys):6d}{ll_of(ys, ps):10.4f}{np.mean((ps >= .5) == ys):9.4f}")

    print("\n=== 同档对局（tier 引擎）的地图差异来源 ===")
    same_p = table["tier"]["p"][table["tier"]["same_idx"]]
    print(f"  同档样本 {len(same_p)} 张地图，平均 |p-0.5| = {np.mean(np.abs(same_p - .5)):.3f}，"
          f"p 范围 [{same_p.min():.3f}, {same_p.max():.3f}]")
    print("  同档时 p 的差异只来自回合净胜残差；veto 决定打哪几张图。")


if __name__ == "__main__":
    main()
