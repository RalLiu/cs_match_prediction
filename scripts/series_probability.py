"""任意两队的 BO1/BO3/BO5 胜率（地图感知）。

算法（详细说明见 doc/PROJECT.md）：
  1) 队伍强度与地图差异分开建模：
         p_m = sigmoid( w_hltv*HLTV特征差 + w_valve*Valve特征差
                        + w_margin*(A在m图的回合残差 − B在m图的回合残差) − c*(A回避−B回避) )
     tier：两榜均分档；points：两榜均使用 log1p 分差。3 个权重 + 1 个回避系数。
     唯一有验证价值的地图差异来源是 round_margin（回合净胜残差）；
     dynamic_map / counterpick 等已证明无增益，不进入胜率。
  2) 系列赛胜率 = 对"会被打到的地图集合"求和：
         P = Σ_set P(set) × P(best-of-k | set 的逐图 p)
     - 已知真实 veto 时：直接把 set 固定为该 3/5 张图。
     - 未知 veto 时给出两个口径：
         a) 规则点估计：沿用 configs/bp_rules.json 的 BP 枚举（情景假设）
         b) 范围：枚举所有合法地图集合（BO1: 7 种, BO3: C(7,3)=35, BO5: C(7,5)=21），
            给出 min/max —— 这个区间宽度就是"地图/veto 到底影响多少"的诚实度量。
  3) 逐图 p 若完全相同，区间会退化为一个点，此时 BO3/BO5 等于二项式闭式解，
     veto 对总体胜率没有任何影响（实测：BO5 对 BP 温度完全不敏感）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_io import key, load_maps, load_rankings, ordinal  # noqa: E402
from map_preference_features import preference  # noqa: E402
from features import avoidance, build_training_features, predict_features  # noqa: E402

CONFIG = ROOT / "configs/bp_rules.json"
POOL = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke"]
# 对称护栏：OOF 实测裁剪 1.3% 样本、LogLoss 代价 0.0000，但挡掉数据里不存在的 0.95。
# 依据：预测 >0.9 的 106 个样本实测只有 0.868；全部 4105 行没有任何分桶超过 0.87。
RAIL = 0.15


# ---------------------------------------------------------------- 分档（tier）
TIER_WIDTH = 80.0     # 档宽基数（积分）
TIER_SLOPE = 0.06     # 档宽随积分增长的比例：宽度 = TIER_WIDTH + TIER_SLOPE * 积分
TIER_DIV = 5.0        # 档差除以该值，仅用于数值缩放
STRENGTH_VERSION = "dual-ranking-v1"
RATING_PROVIDERS = ("hltv", "valve")


def build_tiers(rankings, medians, width=TIER_WIDTH, slope=TIER_SLOPE):
    """HLTV、Valve 各自按本榜最高分独立分档，缺榜用本榜中位数对应档位。"""
    if not math.isfinite(width) or width <= 0 or not math.isfinite(slope) or slope < 0:
        raise ValueError("档宽须为有限正数，档宽斜率须为有限非负数")
    tiers, fallback = {}, {}
    for provider in RATING_PROVIDERS:
        pts = rankings[provider]
        hi = max(pts.values())

        def index(points):
            steps, x = 0, float(points)
            while x < hi:
                x += width + slope * x
                steps += 1
            return steps

        tiers[provider] = {t: index(p) for t, p in pts.items()}
        fallback[provider] = index(medians[provider])
    return tiers, fallback


def tier_features(tiers, fallback):
    """同时替换 HLTV、Valve 两列为各自的档差。"""
    def ratings(r):
        return tuple((tiers[p].get(r["a"], fallback[p])
                      - tiers[p].get(r["b"], fallback[p])) / TIER_DIV
                     for p in RATING_PROVIDERS)
    return ratings


def strength_row(x4, ratings=None):
    """[HLTV 特征差, Valve 特征差, 回合残差]；默认前两列均为 log1p 分差。"""
    hltv, valve = ratings if ratings is not None else x4[:2]
    return np.array([hltv, valve, x4[6]])


# ---------------------------------------------------------------- 训练（3+1 参数）
def fit_strength(rows, ratings=None):
    """ratings(record) 同时给出两榜特征；None 使用基础特征中的双 log1p 分差。"""
    x = np.array([strength_row(r["x4"], ratings(r) if ratings else None) for r in rows])
    scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
    xn = x / scale
    y = np.array([r["y"] for r in rows])
    cnt = Counter((r["date"], r["a"], r["b"]) for r in rows)
    w = np.array([2 ** (-(ordinal(rows[-1]["date"]) - ordinal(r["date"])) / 30)
                  / cnt[(r["date"], r["a"], r["b"])] for r in rows])
    w = w / w.mean()
    av = np.array([r["avoid"] for r in rows])

    def obj(th):
        z = xn @ th[:-1] - th[-1] * av
        return float(np.sum(w * (np.logaddexp(0, z) - y * z))
                     + .01 * th[:-1] @ th[:-1] + .01 * (th[-1] - 1) ** 2)

    def jac(th):
        z = xn @ th[:-1] - th[-1] * av
        e = w * (expit(z) - y)
        return np.r_[xn.T @ e + .02 * th[:-1], -np.sum(av * e) + .02 * (th[-1] - 1)]

    res = minimize(obj, np.r_[np.zeros(3), .3], jac=jac, method="L-BFGS-B",
                   bounds=[(None, None)] * 3 + [(0, 1)])
    return dict(weights=(res.x[:-1] / scale).tolist(), c=float(res.x[-1]))


# ---------------------------------------------------------------- 系列赛数学
def series_win(ps, target):
    """给定实际会打的地图（顺序无关），A 赢下 best-of-k 的概率。"""
    n = len(ps)
    aim = target
    total = 0.0
    for wins_needed in itertools.product([0, 1], repeat=n):
        a_wins = sum(wins_needed)
        if a_wins < aim:
            continue
        p = 1.0
        for p_map, won in zip(ps, wins_needed):
            p *= p_map if won else 1 - p_map
        total += p
    return total


def binomial(p, n, target):
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(target, n + 1))


# ---------------------------------------------------------------- profile（供规则 veto）
MAP_RATE_DAYS = 90  # 计算"全局出图率"的回看窗口


def global_map_rates(rows, as_of, pool, days=MAP_RATE_DAYS):
    """七张图的全局出场占比，作为首ban的先验基线。

    均匀 1/7 是错的：Anubis 近 90 天只占 9.4%、Dust2 占 18.3%。
    用 1/7 会把"该图本来就少见"误当成"该队不打它"。
    """
    now = date.fromisoformat(as_of).toordinal()
    cnt = Counter()
    n = 0
    for r in rows:
        if r["date"] < as_of and now - ordinal(r["date"]) <= days:
            cnt[r["map"]] += 1
            n += 1
    if not n:
        return {m: 1.0 / len(pool) for m in pool}
    return {m: cnt[m] / n for m in pool}


def build_profiles(a, b, as_of, model, rows, pool, rates=None):
    """双方七图的出场统计。额外带上"证据量"，供首ban规则判断是否采信。"""
    rates = rates or global_map_rates(rows, as_of, pool)
    now = date.fromisoformat(as_of).toordinal()
    out = [{}, {}]
    for m in pool:
        for i, t in enumerate([a, b]):
            pref = preference(model["state"].get(t, dict(timeline=[]))["timeline"], m, as_of)
            actual = [r for r in rows if r["map"] == m and t in (r["a"], r["b"])
                      and r["date"] < as_of and now - ordinal(r["date"]) <= 30]
            wins = sum(r["y"] if t == r["a"] else 1 - r["y"] for r in actual)
            n = len(actual)
            out[i][m] = dict(preference=pref, avoidance=avoidance(pref),
                             time_weighted_usage_probability=pref["decayed_share_unshrunk"],
                             usage=pref["recent_count"] / max(1, pref["recent_total"]),
                             recent_maps=n, recent_wins=wins,
                             raw_recent_win_rate=wins / n if n else None,
                             smoothed_recent_win_signal=2 * ((wins + 4) / (n + 8) - .5),
                             decayed_count=pref["decayed_count"],
                             decayed_mass=pref["decayed_mass"],
                             global_rate=rates[m])
    return out


# ---------------------------------------------------------------- 主计算
def predict_pair(team_a, team_b, as_of, pool=POOL, use_rule_veto=True, detail=False,
                 rail=RAIL, fixed_maps=None, engine="tier",
                 tier_width=TIER_WIDTH, tier_slope=TIER_SLOPE):
    obs, names, manifest, raw, quarantine = load_maps()
    rankings, medians, _ = load_rankings()
    rows, current = build_training_features(obs, rankings, medians)
    current.update(names=names, trained_through=rows[-1]["date"])
    a, b = key(team_a), key(team_b)
    if a == b:
        raise ValueError("需要两支不同的队伍")
    known = current["names"]
    if a not in known or b not in known:
        raise ValueError(f"队伍不在数据中: {[x for x in (a, b) if x not in known]}")
    if as_of <= current["trained_through"]:
        raise ValueError(f"日期须晚于训练截止日 {current['trained_through']}")

    tiers, fallback = build_tiers(rankings, medians, tier_width, tier_slope)
    if engine not in ("tier", "points"):
        raise ValueError("engine 必须为 tier 或 points")
    ratings = tier_features(tiers, fallback) if engine == "tier" else None
    model = fit_strength(rows, ratings)
    w, c = model["weights"], model["c"]

    map_p = {}
    missing = []
    for m in pool:
        x, avoid, A, B = predict_features(current, a, b, m, as_of)
        rank_values = ratings(dict(a=a, b=b)) if ratings else None
        z = float(strength_row(x, rank_values) @ np.array(w) - c * avoid)
        raw = expit(z)
        map_p[m] = float(min(max(raw, rail), 1 - rail)) if rail else raw
        for t in (a, b):
            for prov in ("hltv", "valve"):
                if t not in current["rankings"][prov] and t not in missing:
                    missing.append(t)

    result = dict(team_a=known[a], team_b=known[b], as_of=as_of,
                  strength_model=dict(engine=engine, version=STRENGTH_VERSION,
                                      weights=dict(zip(["hltv", "valve", "margin"], w)),
                                      avoidance_c=c,
                                      tier=dict(width=tier_width, slope=tier_slope)
                                      if engine == "tier" else None,
                                      tier_a={p: tiers[p].get(a, fallback[p]) for p in RATING_PROVIDERS}
                                      if engine == "tier" else None,
                                      tier_b={p: tiers[p].get(b, fallback[p]) for p in RATING_PROVIDERS}
                                      if engine == "tier" else None),
                  map_win_probabilities={m: map_p[m] for m in pool},
                  missing_ranking=sorted(set(missing)),
                  formats={})

    # 所有合法地图集合
    all_fmts = ("BO1", "BO3", "BO5")
    if fixed_maps:
        need = {1: "BO1", 3: "BO3", 5: "BO5"}
        if len(fixed_maps) not in need:
            raise ValueError("--maps 需要 1 / 3 / 5 张图")
        fmt = need[len(fixed_maps)]
        sets = {fmt: [tuple(fixed_maps)]}
        all_fmts = (fmt,)
    else:
        sets = {"BO1": [tuple([m]) for m in pool],
                "BO3": list(itertools.combinations(pool, 3)),
                "BO5": list(itertools.combinations(pool, 5))}
    targets = {"BO1": 1, "BO3": 2, "BO5": 3}
    counts = {"BO1": 1, "BO3": 3, "BO5": 5}

    rule = None
    if use_rule_veto and not fixed_maps:
        import json as _json
        from predict_bp import enumerate_paths, summarize_paths, validate_config
        cfg = _json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(cfg)
        profiles = build_profiles(a, b, as_of, current, rows, pool)
        rule = dict(cfg=cfg, profiles=profiles)

    for fmt in all_fmts:
        tgt, k = targets[fmt], counts[fmt]
        vals = [series_win([map_p[m] for m in s], tgt) for s in sets[fmt]]
        lo, hi = min(vals), max(vals)
        best = max(zip(vals, sets[fmt]))
        worst = min(zip(vals, sets[fmt]))
        flat = float(np.mean([map_p[m] for m in pool]))
        entry = dict(
            map_set_count=len(sets[fmt]),
            veto_free_binomial=binomial(flat, k, tgt),
            set_range=[lo, hi],
            best_case_maps=list(best[1]), worst_case_maps=list(worst[1]),
            p_a=None, p_b=None,
        )
        if rule is not None:
            summary = summarize_paths(enumerate_paths(fmt, rule["profiles"], map_p, rule["cfg"]), map_p, 5)
            entry["p_a"] = summary["p_a"]
            entry["p_b"] = summary["p_b"]
            entry["path_count"] = summary["path_count"]
            entry["top_sequences"] = summary["top_sequences"][:3]
            entry["score_distribution"] = summary["score_distribution"]
        result["formats"][fmt] = entry

    if detail:
        result["map_features"] = {m: dict(
            x4=[float(v) for v in predict_features(current, a, b, m, as_of)[0]],
            avoidance=float(predict_features(current, a, b, m, as_of)[1])) for m in pool}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="任意两队的 BO1/BO3/BO5 胜率（地图感知）")
    ap.add_argument("team_a")
    ap.add_argument("team_b")
    ap.add_argument("--date", default="2026-11-18")
    ap.add_argument("--engine", choices=["tier", "points"], default="tier",
                    help="tier=两榜均分档（默认）；points=两榜均取 log1p 分差")
    ap.add_argument("--tier-width", type=float, default=TIER_WIDTH)
    ap.add_argument("--tier-slope", type=float, default=TIER_SLOPE)
    ap.add_argument("--no-veto", action="store_true", help="不跑规则 veto，只给集合范围")
    ap.add_argument("--maps", default="", help="已知会打的地图，逗号分隔（如 Ancient,Dust2,Mirage）")
    ap.add_argument("--rail", type=float, default=RAIL, help=f"对称概率护栏，默认 {RAIL}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    fixed = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    r = predict_pair(args.team_a, args.team_b, args.date, use_rule_veto=not args.no_veto,
                     rail=args.rail, fixed_maps=fixed, engine=args.engine,
                     tier_width=args.tier_width, tier_slope=args.tier_slope)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    A, B = r["team_a"], r["team_b"]
    sm = r["strength_model"]
    print(f"\n{A} vs {B}   as of {r['as_of']}   引擎={sm['engine']}")
    print(f"强度模型: hltv={sm['weights']['hltv']:+.4f}  valve={sm['weights']['valve']:+.4f}  "
          f"margin={sm['weights']['margin']:+.4f}  c={sm['avoidance_c']:.3f}")
    if sm.get("tier") is not None:
        print(f"分档: 两榜分别使用宽度={sm['tier']['width']:.0f}+{sm['tier']['slope']:.0%}×积分")
        for provider in RATING_PROVIDERS:
            ta, tb = sm['tier_a'][provider], sm['tier_b'][provider]
            print(f"  {provider}: {A}=第{ta}档  {B}=第{tb}档  档差={ta - tb:+d}")
    if r["missing_ranking"]:
        print(f"注意：以下队伍缺积分榜，使用对应榜单中位数回退 -> {r['missing_ranking']}")
    print(f"逐图 {A} 胜率: " + "  ".join(f"{m}={v:.3f}" for m, v in r["map_win_probabilities"].items()))
    print(f"地图间极差: {max(r['map_win_probabilities'].values()) - min(r['map_win_probabilities'].values()):.3f}")
    print()
    print(f"{'赛制':<5}{'规则veto点估计':>14}{'地图集范围':>22}{'忽略veto(二项)':>16}")
    for fmt, e in r["formats"].items():
        pa = f"{e['p_a']:.3f}" if e["p_a"] is not None else "—"
        print(f"{fmt:<5}{pa:>14}   [{e['set_range'][0]:.3f}, {e['set_range'][1]:.3f}]{'':>6}"
              f"{e['veto_free_binomial']:>14.3f}")
        print(f"      最有利地图集 {'/'.join(e['best_case_maps'])} | 最不利 {'/'.join(e['worst_case_maps'])}")
    bo3 = r["formats"].get("BO3") or {}
    if bo3.get("top_sequences"):
        print(f"\nBO3 Top1 出图序列: {' → '.join(bo3['top_sequences'][0]['maps'])}"
              f"  概率 {bo3['top_sequences'][0]['probability']:.1%}")


if __name__ == "__main__":
    main()
