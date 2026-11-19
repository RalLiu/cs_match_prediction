"""前向折评估；--compare-margin 对照旧残差/+8 与积分加权净胜/不收缩。

python scripts/evaluate.py --compare-margin --out reports/evaluation/margin_comparison.json
不指定 --out 时只打印，不写文件。
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import load_maps, load_rankings  # noqa: E402
from features import build_training_features  # noqa: E402
from history_weights import MAP_HALF_LIFE_MATCHES  # noqa: E402
from series_probability import (RAIL, STRENGTH_VERSION, build_tiers, fit_strength,  # noqa: E402
                                strength_row, tier_features)

FOLDS = [.40, .55, .70, .85]


def ll_of(y, p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-y * np.log(q) - (1 - y) * np.log1p(-q)))


def metrics(y, p):
    return dict(n=len(y), logloss=ll_of(y, p), brier=float(np.mean((p - y) ** 2)),
                accuracy=float(np.mean((p >= .5) == y)))


def series_cond(p1, p2, p3):
    """已知三图打满时，A 以 2-1 取胜的条件概率。"""
    win = p1*p2*(1-p3) + p1*(1-p2)*p3 + (1-p1)*p2*p3
    lose = (1-p1)*(1-p2)*p3 + (1-p1)*p2*(1-p3) + p1*(1-p2)*(1-p3)
    return win / (win + lose)


def evaluate_model(observations, rankings, medians, engine, margin_method):
    rows, _ = build_training_features(observations, rankings, medians, engine=engine,
                                      margin_method=margin_method)
    dates = sorted({r['date'] for r in rows})
    cuts = [dates[int(len(dates)*q)] for q in FOLDS]
    tiers, fallback = build_tiers(rankings, medians)
    ratings = tier_features(tiers, fallback) if engine == 'tier' else None
    predictions, records, folds = [], [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        training = [r for r in rows if r['date'] < start]
        future = [r for r in rows if start <= r['date'] < end]
        fitted = fit_strength(training, ratings)
        w, c = np.array(fitted['weights']), fitted['c']
        p = np.array([np.clip(1 / (1 + np.exp(-(
            strength_row(r['x4'], ratings(r) if ratings else None) @ w - c*r['avoid']))),
            RAIL, 1-RAIL) for r in future])
        y = np.array([r['y'] for r in future])
        folds.append(dict(start=start, end_exclusive=end, **metrics(y, p)))
        predictions.extend(p)
        records.extend(future)
    p = np.array(predictions)
    y = np.array([r['y'] for r in records])
    level = np.array([(rankings['valve'].get(r['a'], medians['valve'])
                       + rankings['valve'].get(r['b'], medians['valve']))/2 for r in records])
    groups = defaultdict(list)
    for r, probability in zip(records, p):
        groups[(r['date'], r['a'], r['b'])].append((r['y'], probability))
    series_y, series_p = [], []
    for values in groups.values():
        if len(values) == 3:
            series_y.append(float(sum(v[0] for v in values) >= 2))
            series_p.append(series_cond(*[v[1] for v in values]))
    same = np.array([all(tiers[provider].get(r['a'], fallback[provider])
                         == tiers[provider].get(r['b'], fallback[provider])
                         for provider in tiers) for r in records])
    # 保持原三折口径，另评估原先未包含的最后 15% 日期区间。
    hold_training = [r for r in rows if r['date'] < cuts[-1]]
    hold_future = [r for r in rows if r['date'] >= cuts[-1]]
    hold_model = fit_strength(hold_training, ratings)
    hold_w, hold_c = np.array(hold_model['weights']), hold_model['c']
    hold_p = np.array([np.clip(1 / (1 + np.exp(-(
        strength_row(r['x4'], ratings(r) if ratings else None) @ hold_w - hold_c*r['avoid']))),
        RAIL, 1-RAIL) for r in hold_future])
    hold_y = np.array([r['y'] for r in hold_future])
    return dict(engine=engine, margin_method=margin_method, map=metrics(y, p), folds=folds,
                final_holdout=dict(start=cuts[-1], end_inclusive=dates[-1], **metrics(hold_y, hold_p)),
                top1600_logloss=ll_of(y[level >= 1600], p[level >= 1600]),
                top1700_logloss=ll_of(y[level >= 1700], p[level >= 1700]),
                series=metrics(np.array(series_y), np.array(series_p)),
                both_same_tier=dict(n=int(same.sum()),
                    mean_abs_p_minus_half=float(np.mean(np.abs(p[same]-.5))) if same.any() else None),
                cuts=cuts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-margin', action='store_true')
    parser.add_argument('--out', type=Path, help='可选 JSON 结果路径')
    parser.add_argument('--compare-history', type=Path, help='此前版本的评估 JSON，要求输入和切分相同')
    args = parser.parse_args()
    observations, _, manifest, _, _ = load_maps()
    rankings, medians, rank_manifest = load_rankings()
    baseline = None
    if args.compare_history:
        baseline = json.loads(args.compare_history.read_text(encoding='utf-8'))
        hashes = lambda manifests: sorted((r['file'],r['sha256']) for r in manifests)
        if hashes(baseline['source_hashes']) != hashes(manifest+rank_manifest):
            raise ValueError('旧报告与当前输入哈希不一致，不能直接比较')
        if baseline['folds'] != FOLDS:
            raise ValueError('旧报告的评估切分不同，不能直接比较')
    results = []
    methods = ('legacy', 'weighted') if args.compare_margin else ('weighted',)
    for method in methods:
        for engine in ('tier', 'points'):
            result = evaluate_model(observations, rankings, medians, engine, method)
            results.append(result)
            m, s = result['map'], result['series']
            print(f"{method:8s} {engine:6s} | 地图 n={m['n']} LogLoss={m['logloss']:.6f} "
                  f"Brier={m['brier']:.6f} Acc={m['accuracy']:.2%} | "
                  f"三图打满 n={s['n']} LogLoss={s['logloss']:.6f} Acc={s['accuracy']:.2%}", flush=True)
            for fold in result['folds']:
                print(f"  {fold['start']} ~ {fold['end_exclusive']}（不含）: "
                      f"n={fold['n']} LogLoss={fold['logloss']:.6f}", flush=True)
            h = result['final_holdout']
            print(f"  末段留出 {h['start']} ~ {h['end_inclusive']}: n={h['n']} "
                  f"LogLoss={h['logloss']:.6f} Acc={h['accuracy']:.2%}", flush=True)
    if baseline:
        for current in results:
            if current['margin_method'] != 'weighted':
                continue
            previous = next(r for r in baseline['results']
                            if r['engine']==current['engine'] and r['margin_method']=='weighted')
            print(f"{baseline['strength_version']} → {STRENGTH_VERSION} {current['engine']}: "
                  f"地图 LogLoss {previous['map']['logloss']:.6f} → {current['map']['logloss']:.6f}; "
                  f"末段 {previous['final_holdout']['logloss']:.6f} → {current['final_holdout']['logloss']:.6f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(strength_version=STRENGTH_VERSION, folds=FOLDS,
                       margin_half_life_matches=MAP_HALF_LIFE_MATCHES,
                       margin_decay_clock='team_all_maps',
                       training_weights='equal_series_no_calendar_decay',
                       ranking_difficulty='past_dates_only_monotone_logistic_no_calendar_decay',
                       ranking_snapshot_leakage=True, source_hashes=manifest+rank_manifest, results=results)
        if baseline:
            payload['history_baseline'] = dict(strength_version=baseline['strength_version'],
                results=[r for r in baseline['results'] if r['margin_method']=='weighted'])
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已保存：{args.out}')


if __name__ == '__main__':
    main()
