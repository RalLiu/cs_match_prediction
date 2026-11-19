"""赛前状态与特征：HLTV、Valve 均使用 log1p 分差作为基础积分特征。"""
from collections import defaultdict,Counter
import numpy as np
from scipy.special import expit
from map_preference_features import preference
from history_weights import batch_decay
from rating_features import RankingDifficulty, TIER_WIDTH, TIER_SLOPE
FEATURES=['hltv','valve','elo','hltv_missing','valve_missing','dynamic_map','round_margin','counterpick_proxy']

def avoidance(info):
    # 有活动却不打目标图。方向人为约束为不利，但不添加虚构败场。
    return (max(0,1-info['recent_count']/3)*min(1,info['recent_total']/20)*
            min(1,info['other_maps_since_raw']/10))

def summarize(state,map_name,as_of):
    pref=preference(state['timeline'],map_name,as_of)
    v=state['maps'].get(map_name)
    n=v['n'] if v else 0
    dynamic=v['delta']*n/(n+8) if v else 0
    # 新方法只除以场数加权质量；旧残差口径保留 +8 供对照。
    denominator=n+v.get('margin_shrinkage',8) if v else 0
    margin=v['margin']/denominator if denominator>0 else 0
    bad=max(0,-v['result_residual']/(n+8)) if v else 0
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
       np.log1p(point(a,'valve'))-np.log1p(point(b,'valve')),(elo.get(a,1500)-elo.get(b,1500))/400]
    x += [int(a not in ranks[p])-int(b not in ranks[p]) for p in ['hltv','valve']]
    x += [(A['dynamic']-B['dynamic'])/400,A['margin']-B['margin'],A['counterpick']-B['counterpick']]
    return x,A['avoidance']-B['avoidance'],A,B

def form_value(state,as_of):
    values=[]
    for half in [14,60]:
        data=state.get(str(half),dict(n=0,residual=0))
        values.append(data['residual']/(data['n']+8))
    return values[0],values[0]-values[1]

def weighted_margin(net_margin, win_probability):
    """弱队正表现加权更大、负表现更小；双方互换时严格反号。"""
    difficulty=2*(1-win_probability) if net_margin>=0 else 2*win_probability
    return net_margin*difficulty


def build_training_features(observations,rankings,medians,engine='tier',
                            tier_width=TIER_WIDTH,tier_slope=TIER_SLOPE,
                            margin_method='weighted'):
    if margin_method not in ('weighted','legacy'):
        raise ValueError('margin_method 必须为 weighted 或 legacy')
    dates=sorted({r['date'] for r in observations})
    val_date,test_date=dates[int(len(dates)*.70)],dates[int(len(dates)*.85)]
    by_date=defaultdict(list)
    for r in observations:by_date[r['date']].append(r)
    difficulty=RankingDifficulty(rankings,medians,engine,tier_width,tier_slope)
    state=defaultdict(lambda:dict(timeline=[],maps={}))
    elo=defaultdict(lambda:1500.)
    shell=dict(rankings=rankings,medians=medians,elo=elo,state=state)
    rows=[]
    for today in sorted(by_date):
        todays=by_date[today]
        if margin_method=='weighted':difficulty.fit(today)
        for r in todays:
            x,avoid,A,B=predict_features(shell,r['a'],r['b'],r['map'],today)
            split='train' if today<val_date else 'validation' if today<test_date else 'test'
            rows.append(dict(**r,x4=x,avoid=avoid,split=split))
        updates=Counter();team_games=Counter();map_updates=defaultdict(lambda:np.zeros(4))
        for r in todays:
            a,b,y=r['a'],r['b'],r['y'];m=r['map']
            A=summarize(state[a],m,today);B=summarize(state[b],m,today)
            expectation=expit((elo[a]-elo[b])/400*np.log(10))
            map_expectation=expit((elo[a]-elo[b]+A['dynamic']-B['dynamic'])/400*np.log(10))
            margin=(r['sa']-r['sb'])/(r['sa']+r['sb'])
            if margin_method=='weighted':
                adjusted_margin=weighted_margin(margin,difficulty.probability(a,b))
            else:
                expected_margin=2*expit((elo[a]-elo[b])/800)-1
                adjusted_margin=margin-expected_margin
            for t,sign in [(a,1),(b,-1)]:
                team_games[t]+=1
                updates[t]+=24*sign*(y-expectation)
                map_updates[t,m]+=np.array([32*sign*(y-map_expectation),
                    sign*adjusted_margin,sign*(y-expectation),1])
        # 队伍每打一张任意图，所有旧地图历史都推进一次；仅更新本轮实打图。
        for t,count in team_games.items():
            decay,_=batch_decay(count)
            for history in state[t]['maps'].values():
                for field in ('delta','margin','result_residual','n'):
                    history[field]*=decay
        for (t,m),v in map_updates.items():
            old=state[t]['maps'].get(m,dict(delta=0,margin=0,result_residual=0,n=0))
            _,incoming=batch_decay(team_games[t])
            state[t]['maps'][m]=dict(delta=old['delta']+v[0]*incoming,
                margin=old['margin']+v[1]*incoming,
                result_residual=old['result_residual']+v[2]*incoming,n=old['n']+v[3]*incoming,
                margin_shrinkage=0 if margin_method=='weighted' else 8)
        for t,v in updates.items():elo[t]+=v
        for r in todays:
            for t in [r['a'],r['b']]:state[t]['timeline'].append([today,r['map']])
        if margin_method=='weighted':difficulty.update(todays)

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
        for team,vs in residuals.items():
            for half in [14,60]:
                old=form[team].get(str(half),dict(n=0,residual=0))
                decay,incoming=batch_decay(len(vs),half)
                form[team][str(half)]=dict(n=old['n']*decay+len(vs)*incoming,
                    residual=old['residual']*decay+sum(vs)*incoming)

    return rows,dict(rankings=rankings,medians=medians,elo=dict(elo),state=dict(state),
                     form_state=dict(form),margin_method=margin_method)
