"""BP概率和规则正确性检查：python scripts/test_bp.py。"""
import json,math,copy,itertools
from pathlib import Path
from predict_bp import validate_config,enumerate_paths,summarize_paths,series_distribution,series_details,action_probabilities

root=Path(__file__).resolve().parents[1]
config=json.loads((root/'configs/bp_rules.json').read_text(encoding='utf-8'))
validate_config(config)
profile={m:dict(avoidance=0,usage=1/7,smoothed_recent_win_signal=0,
               preference=dict(share_trend=0)) for m in config['maps']}
for fmt,n in [('BO1',1),('BO3',3),('BO5',5)]:
    ps={m:.5 for m in config['maps']}
    paths=enumerate_paths(fmt,[profile,profile],ps,config)
    summary=summarize_paths(paths,ps,10)
    assert len(paths)==10080
    assert math.isclose(summary['p_a'],.5,abs_tol=1e-10)
    assert math.isclose(summary['probability_sum'],1,abs_tol=1e-10)
    assert math.isclose(sum(summary['score_distribution'].values()),1,abs_tol=1e-10)
    assert math.isclose(summary['top_paths_coverage'],10/10080,abs_tol=1e-10)
    for side in [0,1]:assert math.isclose(sum(p['probability'] for p in paths if p['first_actor']==side),.5,abs_tol=1e-10)
    for m in config['maps']:assert math.isclose(summary['selected_map_probability'][m],n/7,abs_tol=1e-10)
    for path in paths:
        assert len(path['maps'])==n and len(set(path['maps']))==n
        actions=path['actions'];assert len({a['map'] for a in actions})==6
        assert all(a['actor']==(path['first_actor']+i)%2 for i,a in enumerate(actions))
        assert [a['action'] for a in actions]==config[fmt]
        assert not {a['map'] for a in actions if a['action']=='ban'}&set(path['maps'])
    expected_played={'BO1':1,'BO3':2.5,'BO5':4.125}[fmt]
    assert math.isclose(sum(summary['actually_played_probability'].values()),expected_played,abs_tol=1e-10)
    if fmt=='BO3':
        ps={m:.7 for m in config['maps']}
        s=summarize_paths(enumerate_paths(fmt,[profile,profile],ps,config),ps,10)
        assert math.isclose(s['p_a'],.784,abs_tol=1e-10)
    if fmt=='BO5':
        ps={m:.7 for m in config['maps']}
        s=summarize_paths(enumerate_paths(fmt,[profile,profile],ps,config),ps,10)
        assert math.isclose(s['p_a'],.83692,abs_tol=1e-10)
example=json.loads((root/'reports/bp/furia__spirit__2026-11-18/prediction.json').read_text(encoding='utf-8'))
for fmt in ['BO1','BO3','BO5']:
    swapped=[example['profiles']['b'],example['profiles']['a']]
    ps={m:1-p for m,p in example['map_win_probabilities_a'].items()}
    reverse=summarize_paths(enumerate_paths(fmt,swapped,ps,config),ps,10)
    assert math.isclose(reverse['p_a']+example['formats'][fmt]['p_a'],1,abs_tol=1e-10)
    assert math.isclose(sum(example['formats'][fmt]['first_ban_probability_a'].values()),1,abs_tol=1e-10)
# 原地图池的最低图优先；被对手移除后恢复正常，不选第二低作为硬规则。
profiles=[copy.deepcopy(profile),copy.deepcopy(profile)]
for side in profiles:
    for i,m in enumerate(config['maps']):side[m]['time_weighted_usage_probability']=i/21
ps={m:.5 for m in config['maps']};lowest=config['maps'][0]
assert action_probabilities(config['maps'],0,'ban',profiles,ps,config,first_ban=True)=={lowest:1.}
remaining=config['maps'][1:]
fallback=action_probabilities(remaining,1,'ban',profiles,ps,config,first_ban=True)
normal=action_probabilities(remaining,1,'ban',profiles,ps,config,first_ban=False)
assert fallback==normal and len(fallback)==6
profiles[0][config['maps'][1]]['time_weighted_usage_probability']=0
tie=action_probabilities(config['maps'],0,'ban',profiles,ps,config,first_ban=True)
assert tie=={config['maps'][0]:.5,config['maps'][1]:.5}
# 首ban 证据门槛：证据足才确定ban；证据不足则摊平到全部剩余图。
def evidenced(mass,zero_map,others):
    p=[copy.deepcopy(profile) for _ in range(2)]
    for side in p:
        for m in config['maps']:
            side[m]['decayed_mass']=mass
            side[m]['decayed_count']=0.0 if m==zero_map else others
            side[m]['global_rate']=1/len(config['maps'])
    return p
strong=evidenced(40.0,config['maps'][0],6.0)
assert action_probabilities(config['maps'],0,'ban',strong,ps,config,first_ban=True)=={config['maps'][0]:1.}
thin=evidenced(0.5,config['maps'][0],0.08)
flat=action_probabilities(config['maps'],0,'ban',thin,ps,config,first_ban=True)
assert len(flat)==len(config['maps'])
assert all(math.isclose(v,1/len(config['maps']),abs_tol=1e-12) for v in flat.values())
# 门槛生效时不能只ban一张：低证据下的动作概率必须"摊平"，不允许出现确定值
assert max(flat.values())<0.99
# 与枚举所有2^N完整独立胜负序列的真值比较，比分在先到胜场时截断。
for ps in [[.23],[.2,.7,.4],[.2,.7,.4,.8,.6]]:
    truth={};target=len(ps)//2+1
    for bits in itertools.product([0,1],repeat=len(ps)):
        mass=math.prod(p if bit else 1-p for p,bit in zip(ps,bits));a=b=0
        for bit in bits:
            a+=bit;b+=1-bit
            if a==target or b==target:break
        label=f'{a}-{b}';truth[label]=truth.get(label,0)+mass
    actual=series_distribution(ps)
    assert truth.keys()==actual.keys()
    assert all(math.isclose(v,actual[k],abs_tol=1e-12) for k,v in truth.items())
_,played=series_details((1.,1.,1.,1.,1.))
assert played==(1,1,1,0,0)
assert series_distribution([1,0,1,0,1])['3-2']==1
for ps in [[.5,.5],[float('nan')],[1.1],[-.1]]:
    try:series_distribution(ps)
    except ValueError:pass
    else:raise AssertionError('Invalid map probabilities accepted')
for maps in [config['maps'][:6],config['maps']+['Overpass'],[config['maps'][0]]*7]:
    bad=copy.deepcopy(config);bad['maps']=maps
    try:validate_config(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid pool accepted')
print('PASS: BO1/BO3/BO5 legal paths; unknown initiative; normalization; analytical truth; early termination; complementary teams; original-pool minimum first ban; normal fallback when already banned; ties; invalid inputs.')
