"""按队伍已打地图场数统计偏好，日期仅用于排序与排除当日/未来记录。"""
from collections import defaultdict

from history_weights import MAP_HALF_LIFE_MATCHES, RECENT_MAPS, batch_decay

FEATURES = ['recent_share_difference', 'share_trend_difference',
            'decayed_share_difference', 'other_maps_since_difference']


def count_window(history, count=RECENT_MAPS):
    """最近 count 场；边界同日完整纳入，避免为同日比赛虚构顺序。"""
    if not history:
        return [], []
    cutoff = history[max(0, len(history)-count)][0]
    return ([r for r in history if r[0] >= cutoff],
            [r for r in history if r[0] < cutoff])


def preference(history, map_name, as_of, map_count=7):
    past = sorted(((d, m) for d, m in history if d < as_of), key=lambda r:r[0])
    recent_rows, older = count_window(past)
    previous_rows, _ = count_window(older)
    recent = [m for _, m in recent_rows]
    previous = [m for _, m in previous_rows]
    n, old_n = len(recent), len(previous)
    share = (recent.count(map_name)+.5)/(n+.5*map_count)
    old_share = (previous.count(map_name)+.5)/(old_n+.5*map_count)
    confidence = n/(n+10)
    trend = (share-old_share)*min(confidence, old_n/(old_n+10))
    by_date = defaultdict(list)
    for d, m in past:
        by_date[d].append(m)
    later = 0
    mass = decayed_count = 0.
    for d in sorted(by_date, reverse=True):
        maps = by_date[d]
        _, incoming = batch_decay(len(maps))
        weight = 2 ** (-later / MAP_HALF_LIFE_MATCHES) * incoming
        mass += weight * len(maps)
        decayed_count += weight * maps.count(map_name)
        later += len(maps)
    ew_share = (decayed_count+.5)/(mass+.5*map_count)
    last = max((d for d, m in past if m == map_name), default=None)
    other = sum(m != map_name and (last is None or d > last) for d, m in past)
    return dict(recent_count=recent.count(map_name), recent_total=n,
                previous_count=previous.count(map_name), previous_total=old_n,
                recent_share=(share-1/map_count)*confidence, share_trend=trend,
                decayed_share=(ew_share-1/map_count)*mass/(mass+10),
                other_maps_since=min(other, 20)/20, other_maps_since_raw=other,
                maps_since_last=sum(d > last for d, _ in past) if last is not None else None,
                recent_share_unshrunk=share, decayed_share_unshrunk=ew_share,
                decayed_count=decayed_count, decayed_mass=mass)


def differences(a, b):
    return dict(zip(FEATURES, [a[k]-b[k] for k in
                              ['recent_share', 'share_trend', 'decayed_share', 'other_maps_since']]))
