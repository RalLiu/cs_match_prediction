"""BO1/BO3/BO5偏好假设BP枚举与逐图胜率积分。先手固定未知，各50%。

逐图胜率来自3特征强度模型（series_probability.fit_strength，4个参数，每次运行重拟合），
不再依赖 reports/extended_v5/model.joblib。BO/V5工件仅作历史对照保留。
"""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import argparse,csv,hashlib,itertools,json,math,re
import numpy as np
from scipy.special import expit
from features import build_training_features,predict_features
from data_io import load_maps,load_rankings,key
from history_weights import MAP_HALF_LIFE_MATCHES
from series_probability import (RAIL,RATING_PROVIDERS,STRENGTH_VERSION,TIER_SLOPE,TIER_WIDTH,
                                binomial,build_profiles,build_tiers,
                                fit_strength,series_win,strength_row,tier_features)

ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=8192)
def series_details(ps):
    """逐图递推，到达所需胜场立即结束，返回终局比分与各图实际开打概率。"""
    if len(ps) not in [1,3,5] or any(not math.isfinite(p) or not 0<=p<=1 for p in ps):
        raise ValueError('需要1/3/5个有限的[0,1]地图胜率')
    target=len(ps)//2+1;states={(0,0):1.};outcomes=defaultdict(float);played=[]
    for p in ps:
        played.append(sum(states.values()));next_states=defaultdict(float)
        for (a,b),mass in states.items():
            for na,nb,prob in [(a+1,b,p),(a,b+1,1-p)]:
                if na==target or nb==target:outcomes[f'{na}-{nb}']+=mass*prob
                else:next_states[na,nb]+=mass*prob
        states=next_states
    return dict(outcomes),tuple(played)

def series_distribution(ps):
    return dict(series_details(tuple(ps))[0])

def validate_config(config):
    if len(config['maps'])!=7 or len(set(config['maps']))!=7:raise ValueError('配置必须有七张不同地图')
    if config['BO1']!=['ban']*6:raise ValueError('当前引擎BO1只支持交替六ban')
    if config['BO3']!=['ban','ban','pick','pick','ban','ban']:raise ValueError('当前引擎BO3顺序为ban ban pick pick ban ban')
    if config.get('BO5')!=['ban','ban','pick','pick','pick','pick']:raise ValueError('当前引擎BO5顺序为ban ban pick pick pick pick')
    if config['first_actor']!='unknown_equal_mixture':raise ValueError('本版本固定先手未知，各50%')
    if config.get('first_ban_rule')!='lowest_usage_unless_already_banned':raise ValueError('首ban规则须为最低出场图优先，已被ban则恢复综合评分')
    if not math.isfinite(config['temperature']) or config['temperature']<=0:raise ValueError('temperature必须为有限正数')
    if any(not math.isfinite(v) or v<0 for v in config['weights'].values()):raise ValueError('偏好权重必须有限非负')

FIRST_BAN_KAPPA=2.0          # 先验强度：相当于看了 2 张均匀分布的图
FIRST_BAN_MIN_EXPECTED=2.0   # 采信门槛：按全局出图率，该图"应该"至少出场这么多次

def first_ban_analysis(own,config,remaining=None):
    """首ban 的依据与证据强度。

    先验不用均匀 1/7，而用最近约 500 张历史地图的全局出图率：
        share(m) = (加权出场 + kappa*全局率(m)) / (加权总质量 + kappa)
    并给出 expected = 质量 * 全局率(m)：若正常打这张图"应该见到的次数"。
    expected < min_expected 时，零出场更可能是环境原因，不足以判定为ban。
    """
    pool=config['maps'];n=len(pool)
    remaining=list(pool if remaining is None else remaining)
    kappa=float(config.get('first_ban_kappa',FIRST_BAN_KAPPA))
    min_expected=float(config.get('first_ban_min_expected',FIRST_BAN_MIN_EXPECTED))
    prior=config.get('first_ban_prior','global_map_rate_last500')
    shares={};evidence={}
    for m in pool:
        e=own[m]
        cnt=e.get('decayed_count');mass=e.get('decayed_mass')
        rate=e.get('global_rate')
        if prior=='uniform' or cnt is None or mass is None:
            # 旧口径：均匀 1/7 先验、无证据门槛。保留以便回溯与对照。
            shares[m]=e.get('count_weighted_usage_probability',e.get('time_weighted_usage_probability',e['usage']))
            evidence[m]=None
            continue
        rate=1.0/n if rate is None else rate
        shares[m]=(cnt+kappa*rate)/(mass+kappa)
        evidence[m]=dict(decayed_count=round(cnt,3),decayed_mass=round(mass,3),
            global_rate=round(rate,4),expected=round(mass*rate,3))
    lowest=min(shares.values())
    tied=[m for m in pool if math.isclose(shares[m],lowest,rel_tol=0,abs_tol=1e-12)]
    candidates=[m for m in remaining if m in tied]
    audit=dict(lowest_map=tied[0],lowest_share=round(lowest,4),tied=tied,
               remaining_candidates=candidates,prior=prior,
               evidence=evidence[tied[0]],
               min_expected=min_expected,kappa=kappa)
    if not candidates:
        audit['decision']='fallback_composite'   # 原最低图已被对手 ban
        return None,audit
    gated=[c for c in candidates
           if evidence[c] is None or evidence[c]['expected']>=min_expected]
    if evidence[candidates[0]] is not None and not gated:
        audit['decision']='insufficient_evidence'  # 证据不足：均摊到剩余图，不假装知道
        return {m:1/len(remaining) for m in remaining},audit
    audit['decision']='rule' if evidence[candidates[0]] is not None else 'rule_legacy'
    return {m:1/len(gated) for m in gated},audit

def first_ban_audit(profiles,config):
    """双方各自的首ban 判定与证据，供输出与报告使用（在完整七图上评估）。"""
    return {'a':first_ban_analysis(profiles[0],config)[1],
            'b':first_ban_analysis(profiles[1],config)[1]}

def first_ban_sentence(audit,name_a,name_b):
    """把首ban 的依据与证据写成一句话，放进 report.md。"""
    parts=[]
    for side,name in (('a',name_a),('b',name_b)):
        x=audit[side];ev=x.get('evidence') or {}
        if x['decision']=='insufficient_evidence':
            parts.append(f"{name} 证据不足（最低图 {x['lowest_map']} 加权出场 {ev.get('decayed_count')}、"
                         f"有效样本量 {ev.get('decayed_mass')}，按全局出图率本该出场 {ev.get('expected')} 次），"
                         f"改为在剩余图上均分")
            continue
        head=f"{name} 首ban {x['lowest_map']}"
        if ev:
            head+=(f"（该图按已打地图场数加权出场 {ev['decayed_count']}、有效样本量 {ev['decayed_mass']}；"
                   f"按全局出图率 {ev['global_rate']:.1%} 本该出场 {ev['expected']} 次）")
        if x['decision']=='fallback_composite':
            head+="，该图已被对手ban，回落综合评分"
        parts.append(head)
    return "首ban 依据：" + "；".join(parts) + "。"

def action_probabilities(remaining,actor,action,profiles,map_probs,config,first_ban=False):
    if first_ban and action=='ban':
        # 最低值从原始七图池确定，不在剩余图上重新寻找第二低值。
        probs,_=first_ban_analysis(profiles[actor],config,remaining)
        if probs is not None:return probs
        # 原最低出场图已被对手ban，继续下方原综合评分。
    scores=[];weights=config['weights']
    for m in remaining:
        own,opp=profiles[actor][m],profiles[1-actor][m]
        p=map_probs[m] if actor==0 else 1-map_probs[m]
        # V5已包含实力/回避，动作偏好另描述是否愿意让此图出现。
        # 它们不是独立证据，参数只作为情景，不声称经过行为训练。
        if action=='ban':
            value=(weights['ban_avoidance']*own['avoidance']+
                weights['ban_opponent_usage']*opp['usage']*(1-p)+
                weights['ban_matchup_risk']*math.log((1-p)/p))
        else:
            value=(weights['pick_usage']*own['usage']+
                weights['pick_matchup']*math.log(p/(1-p))+
                weights['pick_recent_form']*own['smoothed_recent_win_signal']+
                weights['pick_trend']*own['preference']['share_trend'])
        scores.append(value/config['temperature'])
    values=np.array(scores);values-=values.max();p=np.exp(values);p/=p.sum()
    return dict(zip(remaining,p.tolist()))

def enumerate_paths(format_name,profiles,map_probs,config):
    paths=[];steps=config[format_name]
    for first in [0,1]:
        def walk(remaining,step,prob,actions,picks):
            if step==len(steps):
                assert len(remaining)==1
                maps=picks+[remaining[0]]
                ps=[map_probs[m] for m in maps];distribution=series_distribution(ps)
                pa=sum(v for score,v in distribution.items() if int(score.split('-')[0])>int(score.split('-')[1]))
                paths.append(dict(first_actor=first,probability=prob,actions=actions,
                    maps=maps,conditional_p_a=pa,score_distribution=distribution))
                return
            actor=first if step%2==0 else 1-first;action=steps[step]
            for m,p in action_probabilities(remaining,actor,action,profiles,map_probs,config,first_ban=step<2).items():
                walk([v for v in remaining if v!=m],step+1,prob*p,
                    actions+[dict(actor=actor,action=action,map=m,conditional_probability=p)],
                    picks+[m] if action=='pick' else picks)
        walk(config['maps'],0,.5,[],[])
    return paths

def summarize_paths(paths,map_probs,top):
    paths=sorted(paths,key=lambda r:r['probability'],reverse=True)
    sequences=defaultdict(float);scores=defaultdict(float)
    selected=defaultdict(float);played=defaultdict(float);firstban=[defaultdict(float),defaultdict(float)]
    for r in paths:
        p=r['probability'];maps=r['maps'];sequences[tuple(maps)]+=p
        for s,v in r['score_distribution'].items():scores[s]+=p*v
        for m in maps:selected[m]+=p
        _,reach=series_details(tuple(map_probs[m] for m in maps))
        for m,rp in zip(maps,reach):played[m]+=p*rp
        for a in r['actions'][:2]:firstban[a['actor']][a['map']]+=p
    ranked=sorted(sequences.items(),key=lambda pair:pair[1],reverse=True)
    top_paths=paths[:top];top_sequences=[dict(maps=list(ms),probability=p) for ms,p in ranked[:top]]
    coverage=sum(r['probability'] for r in top_paths)
    pa=sum(r['probability']*r['conditional_p_a'] for r in paths)
    return dict(path_count=len(paths),probability_sum=sum(r['probability'] for r in paths),p_a=pa,p_b=1-pa,
        score_distribution=dict(scores),top_paths=top_paths,top_paths_coverage=coverage,
        top_sequences=top_sequences,top_sequences_coverage=sum(r['probability'] for r in top_sequences),
        selected_map_probability=dict(selected),actually_played_probability=dict(played),
        first_ban_probability_a=dict(firstban[0]),first_ban_probability_b=dict(firstban[1]))

SNAPSHOT_DATE='2026-11-18'  # 两张积分榜快照中较晚者；早于此日期无法从当期快照预测

def prepare(a,b,as_of,config,engine='tier',tier_width=TIER_WIDTH,tier_slope=TIER_SLOPE):
    """逐图胜率 = 3特征强度模型；state/profile 由当前CSV现场重建，不读工件。"""
    observations,names,manifest,raw,quarantine=load_maps()
    rankings,medians,rank_manifest=load_rankings()
    if a==b or a not in names or b not in names:raise ValueError('需要两个不同的已知队伍')
    rows,model=build_training_features(observations,rankings,medians,engine=engine,
        tier_width=tier_width,tier_slope=tier_slope)
    model.update(names=names,trained_through=rows[-1]['date'])
    if as_of<=model['trained_through'] or as_of<SNAPSHOT_DATE:
        raise ValueError(f"预测日期需晚于数据截止日{model['trained_through']}且不早于积分快照{SNAPSHOT_DATE}")
    known={m for s in model['state'].values() for m in s['maps']}
    if not set(config['maps'])<=known:raise ValueError('地图池包含数据中未出现的地图')
    tiers,fallback=build_tiers(rankings,medians,tier_width,tier_slope)
    if engine not in ('tier','points'):raise ValueError('engine 必须为 tier 或 points')
    ratings=tier_features(tiers,fallback) if engine=='tier' else None
    strength=fit_strength(rows,ratings)
    w,c=strength['weights'],strength['c']
    rank_values=ratings(dict(a=a,b=b)) if ratings else None
    map_probs={}
    for m in config['maps']:
        x,av,A,B=predict_features(model,a,b,m,as_of)
        raw_p=float(expit(strength_row(x,rank_values)@np.array(w)-c*av))
        map_probs[m]=float(min(max(raw_p,RAIL),1-RAIL))
    profiles=build_profiles(a,b,as_of,model,rows,config['maps'])
    provenance=dict(engine=engine,strength_version=STRENGTH_VERSION,margin_method=model['margin_method'],
        strength=dict(weights=dict(zip(['hltv','valve','margin'],w)),avoidance_c=c,rail=RAIL,
                      tier=dict(width=tier_width,slope=tier_slope) if engine=='tier' else None,
                      tier_a={p:tiers[p].get(a,fallback[p]) for p in RATING_PROVIDERS} if engine=='tier' else None,
                      tier_b={p:tiers[p].get(b,fallback[p]) for p in RATING_PROVIDERS} if engine=='tier' else None),
        n_maps=len(rows),trained_through=model['trained_through'],
        missing_ranking=sorted({t for t in (a,b) for p in ('hltv','valve') if t not in rankings[p]}),
        source_hashes=manifest+rank_manifest)
    return model['names'][a],model['names'][b],map_probs,profiles,provenance

def map_set_range(map_probs,pool,fmt):
    """不依赖任何veto权重的地图集合范围：所有合法集合下的min/max与二项对照。"""
    k,target={'BO1':(1,1),'BO3':(3,2),'BO5':(5,3)}[fmt]
    lo,hi,best,worst=1.0,0.0,None,None
    for s in itertools.combinations(pool,k):
        v=series_win([map_probs[m] for m in s],target)
        if v>hi:hi,best=v,s
        if v<lo:lo,worst=v,s
    flat=float(np.mean([map_probs[m] for m in pool]))
    return dict(set_range=[lo,hi],best_case_maps=list(best),worst_case_maps=list(worst),
        set_count=len(list(itertools.combinations(pool,k))),veto_free_binomial=binomial(flat,k,target))

def main():
    parser=argparse.ArgumentParser(description='BP偏好情景＋强度模型整场胜率，BO1/BO3/BO5均默认先手未知')
    parser.add_argument('team_a');parser.add_argument('team_b');parser.add_argument('--format',choices=['BO1','BO3','BO5','both','all'],default='both',help='both保留BO1/BO3；all计算三种')
    parser.add_argument('--date',default='2026-11-18');parser.add_argument('--top',type=int,default=10)
    parser.add_argument('--config',default=str(ROOT/'configs/bp_rules.json'))
    parser.add_argument('--out',default='',help='输出根目录，默认 reports/bp（用于对照实验时写临时目录）')
    parser.add_argument('--engine',choices=['tier','points'],default='tier',help='tier=两榜均分档（默认）；points=两榜均取 log1p 分差')
    parser.add_argument('--tier-width',type=float,default=TIER_WIDTH)
    parser.add_argument('--tier-slope',type=float,default=TIER_SLOPE)
    args=parser.parse_args()
    if args.top<1:parser.error('top必须为正数')
    try:
        config=json.loads(Path(args.config).read_text(encoding='utf-8'));validate_config(config)
        a,b=key(args.team_a),key(args.team_b)
        name_a,name_b,map_probs,profiles,provenance=prepare(a,b,args.date,config,engine=args.engine,
            tier_width=args.tier_width,tier_slope=args.tier_slope)
    except (ValueError,KeyError) as e:parser.error(str(e))
    safe=lambda s:re.sub(r'[^\w.-]+','_',s)
    root=Path(args.out) if args.out else ROOT/'reports/bp'
    out=root/f'{safe(a)}__{safe(b)}__{args.date}';out.mkdir(parents=True,exist_ok=True)
    result=dict(team_a=name_a,team_b=name_b,as_of=args.date,first_actor='unknown_50_50',
        maps=config['maps'],map_win_probabilities_a=map_probs,profiles=dict(a=profiles[0],b=profiles[1]),config=config,
        provenance=provenance,first_ban=first_ban_audit(profiles,config),
        warning='单图胜率来自3特征强度模型（4参数），使用[0.15,0.85]逐图护栏。'
                'tier 对 HLTV/Valve 均分档，points 对两榜均取 log1p 分差。'
                'BO5 会放大单图概率误差；地图集合范围只表示地图选择的敏感性，不是置信区间。'
                'BP动作概率为手工偏好情景，未经真实Veto标签训练，赛制协议仍需按赛事确认。',formats={})
    formats=['BO1','BO3','BO5'] if args.format=='all' else ['BO1','BO3'] if args.format=='both' else [args.format]
    for fmt in formats:
        paths=enumerate_paths(fmt,profiles,map_probs,config);summary=summarize_paths(paths,map_probs,args.top)
        assert math.isclose(summary['probability_sum'],1,abs_tol=1e-10)
        assert math.isclose(sum(summary['score_distribution'].values()),1,abs_tol=1e-10)
        summary.update(map_set_range(map_probs,config['maps'],fmt))
        result['formats'][fmt]=summary
        sensitivity=[]
        for temperature in [.7,1.5]:
            scenario={**config,'temperature':temperature}
            sp=enumerate_paths(fmt,profiles,map_probs,scenario)
            sensitivity.append(dict(temperature=temperature,p_a=sum(r['probability']*r['conditional_p_a'] for r in sp)))
        summary['temperature_sensitivity']=sensitivity
        with (out/f'{fmt}_all_paths.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['first_actor','probability','actions','maps','conditional_p_a','weighted_p_a']);writer.writeheader()
            for p in sorted(paths,key=lambda r:r['probability'],reverse=True):writer.writerow(dict(first_actor='a' if p['first_actor']==0 else 'b',
                probability=p['probability'],actions=json.dumps(p['actions'],ensure_ascii=False),maps=' / '.join(p['maps']),
                conditional_p_a=p['conditional_p_a'],weighted_p_a=p['probability']*p['conditional_p_a']))
    result['config_sha256']=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    (out/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[f'# {name_a} 对 {name_b}：BP与整场胜率情景','',result['warning'],'',
        '全部情况先手未知，A先/B先各50%。BO1交替六ban；BO3 ban-ban-pick-pick-ban-ban；BO5 ban-ban-pick-pick-pick-pick。均以剩余地图决胜。',
        f"胜率引擎：{args.engine}，模型版本：{STRENGTH_VERSION}。"
        + ('HLTV 与 Valve 分别分档，各取档差/5。' if args.engine=='tier'
           else 'HLTV 与 Valve 均取 log(1+积分) 之差。')
        + f"拟合权重：{provenance['strength']['weights']}；回避系数：{provenance['strength']['avoidance_c']:.4f}。"
        + '逐图 logit 为两榜特征加权和 + 地图积分加权净胜项 − 回避项，再截断到[0.15,0.85]。'
        + f'地图净胜按双方积分难度加权，以该队之后任意地图{MAP_HALF_LIFE_MATCHES}场为半衰期，不做样本量收缩；日期只用于历史顺序和排除未来赛果。',
        '偏好基于实际出场而非真实pick记录；ban考虑自身回避/对手常用/对阵风险，pick考虑自身常用/近期趋势/平滑胜率及强度模型对阵胜率。softmax只在剩余图中分配动作概率。',
        '双方首ban取原始地图池中"加权出场占比相对全局出图率最低"的图（先验不是均匀 1/7），并列则均分；该图若已被对手ban，则恢复原综合评分。若按全局出图率该图本该出场不足 2 次，则判为证据不足，改为在剩余图上均分。',
        first_ban_sentence(result['first_ban'],name_a,name_b),
        'BP权重和温度为人工设置，尚未训练，也没有真实BP回测；不能把“路径概率20%”理解为已校准的实际发生频率。动作偏好与胜率模型存在重叠信息，不能视为独立证据。',
        '当前评分逐步选择，没有博弈最优或全局后向规划；先交付可解释基线，真实Veto数据到位后才能验证行为。','']
    for fmt,s in result['formats'].items():
        lines += [f'## {fmt}', '',f"{name_a}胜率：{s['p_a']:.2%}；{name_b}：{s['p_b']:.2%}。",
            f"veto 影响范围：{s['set_range'][0]:.2%} ~ {s['set_range'][1]:.2%}"
            f"（{s['set_count']}种合法地图集；最有利 {'/'.join(s['best_case_maps'])}，"
            f"最不利 {'/'.join(s['worst_case_maps'])}）；忽略veto的二项对照{s['veto_free_binomial']:.2%}。",
            f"精确枚举{s['path_count']}条路径，概率和{s['probability_sum']:.12f}。整场胜率使用全部路径，Top {args.top}路径覆盖{s['top_paths_coverage']:.2%}。",'',
            f'概率最高的地图组合（合并不同ban顺序；顺序表示实际出图次序）：','', '|地图/序列|概率|','|---|---:|']
        for r in s['top_sequences']:lines.append(f"|{' → '.join(r['maps'])}|{r['probability']:.2%}|")
        lines += ['', '概率最高的完整动作路径：','', '|先手|动作|路径概率|该路径A胜率|','|---|---|---:|---:|']
        for r in s['top_paths']:
            acts='；'.join(f"{name_a if act['actor']==0 else name_b} {act['action']} {act['map']}" for act in r['actions'])
            lines.append(f"|{name_a if r['first_actor']==0 else name_b}|{acts}|{r['probability']:.2%}|{r['conditional_p_a']:.2%}|")
        lines += ['',f"比分分布：{s['score_distribution']}。",f"动作温度敏感性：{s['temperature_sensitivity']}。仅改变选择集中程度，不是置信区间。",'']
    lines += ['每条BP路径按逐图独立近似递推胜负，到先赢1/2/3图即停止，分别得到BO1/BO3/BO5胜率及终局比分，再按全部路径概率求和。不假设2:0后还打第三图，也不假设3:0后还打第四/第五图。无阵营选择或同场状态相关。',
        '地图集合范围不依赖任何veto权重：枚举全部合法集合（BO1 7种、BO3 35种、BO5 21种）取min/max，代表“无论veto怎么走”的边界。',
        '若七张图的胜率完全相同，集合范围退化为单点，此时veto对总胜率严格为0——这是数学必然，不是缺陷。',
        'selected_map_probability表示被选入地图序列，不代表一定实际打过；actually_played_probability根据系列赛仍未结束的概率累计。',
        '完整动作Top与覆盖率见prediction.json，全部路径见各赛制_all_paths.csv。0/1代表输入队A/B，first_actor表示谁先行动。已知真实BP后应改为相应条件预测，而非继续混合全部路径。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(team_a=name_a,team_b=name_b,first_actor='unknown_50_50',
        output=str(out),formats={f:dict(p_a=s['p_a'],p_b=s['p_b'],paths=s['path_count'],
            set_range=s['set_range'],veto_free_binomial=s['veto_free_binomial'],
            top_paths_coverage=s['top_paths_coverage'],top_sequences=s['top_sequences'][:3]) for f,s in result['formats'].items()}),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
