"""仅使用切点之前日期的实际地图出场记录，不能据此认定为主动pick。"""
from datetime import date
import math

FEATURES=['recent_share_difference','share_trend_difference',
          'decayed_share_difference','other_maps_since_difference']

def preference(history,map_name,as_of,map_count=7):
    """history为(date,map)序列；同日全部排除，不对同系列赛逐图更新。"""
    end=date.fromisoformat(as_of).toordinal()
    past=[(end-date.fromisoformat(d).toordinal(),m) for d,m in history
          if date.fromisoformat(d).toordinal()<end]
    recent=[m for age,m in past if age<=30]
    previous=[m for age,m in past if 30<age<=60]
    n,old_n=len(recent),len(previous)
    share=(recent.count(map_name)+.5)/(n+.5*map_count)
    old_share=(previous.count(map_name)+.5)/(old_n+.5*map_count)
    # 没有比赛时，不能仅因时间过去就推断弃图；低活动量降低权重。
    confidence=n/(n+10)
    trend=(share-old_share)*min(confidence,old_n/(old_n+10))
    weights=[(2**(-age/21),m) for age,m in past]
    mass=sum(w for w,m in weights)
    decayed_count=sum(w for w,m in weights if m==map_name)
    # ew_share 用均匀 1/map_count 做先验；首ban 判定另用全局出图率做先验（见 predict_bp）。
    ew_share=(decayed_count+.5)/(mass+.5*map_count)
    age_last=min((age for age,m in past if m==map_name),default=None)
    # 距上次打目标图后实际打过多少其他图；区别停赛与主动回避。
    other=sum(m!=map_name and (age_last is None or age<age_last) for age,m in past)
    return dict(recent_count=recent.count(map_name),recent_total=n,
        previous_count=previous.count(map_name),previous_total=old_n,
        recent_share=(share-1/map_count)*confidence,share_trend=trend,
        decayed_share=(ew_share-1/map_count)*mass/(mass+10),
        other_maps_since=min(other,20)/20,other_maps_since_raw=other,days_since_last=age_last,
        recent_share_unshrunk=share,decayed_share_unshrunk=ew_share,
        decayed_count=decayed_count,decayed_mass=mass)

def differences(a,b):
    return dict(zip(FEATURES,[a[k]-b[k] for k in
        ['recent_share','share_trend','decayed_share','other_maps_since']]))
