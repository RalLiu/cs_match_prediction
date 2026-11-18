"""V5固定的赛前特征与时间更新规则。保留现有模型计算方式。"""
from collections import defaultdict,Counter
from datetime import date
import numpy as np
from scipy.special import expit
from map_preference_features import preference
from data_io import ordinal
HALF=30
FEATURES=['hltv','valve','elo','hltv_missing','valve_missing','dynamic_map','round_margin','counterpick_proxy']

def avoidance(info):
    # 有活动却不打目标图。方向人为约束为不利，但不添加虚构败场。
    return (max(0,1-info['recent_count']/3)*min(1,info['recent_total']/20)*
            min(1,info['other_maps_since_raw']/10))

def summarize(state,map_name,as_of):
    now=date.fromisoformat(as_of).toordinal()
    pref=preference(state['timeline'],map_name,as_of)
    v=state['maps'].get(map_name)
    f=2**(-(now-v['last'])/HALF) if v else 0
    n=v['n']*f if v else 0
    dynamic=v['delta']*f*n/(n+8) if v else 0
    margin=v['margin']*f/(n+8) if v else 0
    bad=max(0,-v['result_residual']*f/(n+8)) if v else 0
    # 多打但低于对手实力预期的结果：仅是疑似被针对，不是真实pick。
    exposure=pref['recent_count']/max(1,pref['recent_total'])
    return dict(preference=pref,n=n,dynamic=dynamic,margin=margin,
                counterpick=exposure*bad,avoidance=avoidance(pref))

def predict_features(model,a,b,map_name,as_of):
    ranks=model['rankings'];med=model['medians'];elo=model['elo']
    def point(t,p):return ranks[p].get(t,med[p])
    empty=dict(timeline=[],maps={})
    A=summarize(model['state'].get(a,empty),map_name,as_of)
    B=summarize(model['state'].get(b,empty),map_name,as_of)
    x=[np.log1p(point(a,'hltv'))-np.log1p(point(b,'hltv')),
       (point(a,'valve')-point(b,'valve'))/400,(elo.get(a,1500)-elo.get(b,1500))/400]
    x += [int(a not in ranks[p])-int(b not in ranks[p]) for p in ['hltv','valve']]
    x += [(A['dynamic']-B['dynamic'])/400,A['margin']-B['margin'],A['counterpick']-B['counterpick']]
    return x,A['avoidance']-B['avoidance'],A,B

def form_value(state,as_of):
    now=date.fromisoformat(as_of).toordinal()
    values=[]
    for half in [14,60]:
        data=state.get(str(half),dict(n=0,residual=0,last=now))
        f=2**(-(now-data['last'])/half)
        values.append(data['residual']*f/(data['n']*f+8))
    return values[0],values[0]-values[1]

def build_training_features(observations,rankings,medians):
    dates=sorted({r['date'] for r in observations})
    val_date,test_date=dates[int(len(dates)*.70)],dates[int(len(dates)*.85)]
    by_date=defaultdict(list)
    for r in observations:by_date[r['date']].append(r)
    state=defaultdict(lambda:dict(timeline=[],maps={}))
    elo=defaultdict(lambda:1500.)
    shell=dict(rankings=rankings,medians=medians,elo=elo,state=state)
    rows=[]
    for today,todays in by_date.items():
        now=ordinal(today)
        for r in todays:
            x,avoid,A,B=predict_features(shell,r['a'],r['b'],r['map'],today)
            split='train' if today<val_date else 'validation' if today<test_date else 'test'
            rows.append(dict(**r,x4=x,avoid=avoid,split=split))
        updates=Counter();map_updates=defaultdict(lambda:np.zeros(4))
        for r in todays:
            a,b,y=r['a'],r['b'],r['y'];m=r['map']
            A=summarize(state[a],m,today);B=summarize(state[b],m,today)
            expectation=expit((elo[a]-elo[b])/400*np.log(10))
            map_expectation=expit((elo[a]-elo[b]+A['dynamic']-B['dynamic'])/400*np.log(10))
            margin=(r['sa']-r['sb'])/(r['sa']+r['sb'])
            # 预期回合净胜比例的平滑近似，仅作工程特征，由留出指标验证。
            expected_margin=2*expit((elo[a]-elo[b])/800)-1
            for t,sign in [(a,1),(b,-1)]:
                updates[t]+=24*sign*(y-expectation)
                map_updates[t,m]+=np.array([32*sign*(y-map_expectation),
                    sign*(margin-expected_margin),sign*(y-expectation),1])
        for (t,m),v in map_updates.items():
            old=state[t]['maps'].get(m,dict(delta=0,margin=0,result_residual=0,n=0,last=now))
            f=2**(-(now-old['last'])/HALF)
            state[t]['maps'][m]=dict(delta=old['delta']*f+v[0],margin=old['margin']*f+v[1],
                result_residual=old['result_residual']*f+v[2],n=old['n']*f+v[3],last=now)
        for t,v in updates.items():elo[t]+=v
        for r in todays:
            for t in [r['a'],r['b']]:state[t]['timeline'].append([today,r['map']])

    form=defaultdict(dict)
    by_date=defaultdict(list)
    for r in rows:
        by_date[r['date']].append(r)
    for today,todays in by_date.items():
        for r in todays:
            a=form_value(form[r['a']],today);b=form_value(form[r['b']],today)
            r['x4'] += [a[0]-b[0],a[1]-b[1]]
        residuals=defaultdict(list)
        for r in todays:
            expected=expit(r['x4'][2]*np.log(10))
            residuals[r['a']].append(r['y']-expected)
            residuals[r['b']].append(expected-r['y'])
        now=ordinal(today)
        for team,vs in residuals.items():
            for half in [14,60]:
                old=form[team].get(str(half),dict(n=0,residual=0,last=now))
                decay=2**(-(now-old['last'])/half)
                form[team][str(half)]=dict(n=old['n']*decay+len(vs),
                    residual=old['residual']*decay+sum(vs),last=now)

    return rows,dict(rankings=rankings,medians=medians,elo=dict(elo),state=dict(state),form_state=dict(form))
