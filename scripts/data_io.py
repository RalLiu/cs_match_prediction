"""比赛/积分快照CSV读取、精确镜像合并与输入清单。"""
from pathlib import Path
from datetime import date
import csv,hashlib,unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
key=lambda s:unicodedata.normalize('NFKC',s).strip().lower()
ordinal=lambda s:date.fromisoformat(s).toordinal()

def load_maps():
    groups={};names={};manifest=[];raw=0
    for file in sorted((ROOT/'data/team_maps').glob('*.csv')):
        manifest.append(dict(file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        with file.open(encoding='utf-8-sig',newline='') as f:
            for line,r in enumerate(csv.DictReader(f),2):
                raw+=1;a,b=key(r['team']),key(r['opponent']);sa,sb=int(r['team_score']),int(r['opponent_score'])
                names[a]=r['team'];names.setdefault(b,r['opponent']);date.fromisoformat(r['date'])
                assert a!=b and min(sa,sb)>=0 and sa!=sb and r['result']==('W' if sa>sb else 'L'),(file,line)
                reverse=a>b
                if reverse:a,b,sa,sb=b,a,sb,sa
                identity=(r['date'],r['event'],r['map'],a,b,sa,sb)
                g=groups.setdefault(identity,dict(date=r['date'],event=r['event'],map=r['map'],a=a,b=b,
                    sa=sa,sb=sb,y=int(sa>sb),sides=[0,0],refs=[]))
                g['sides'][int(reverse)]+=1;g['refs'].append(dict(file=file.name,row=line))
    quarantine=[r for r in groups.values() if max(r['sides'])>1]
    rows=sorted([r for r in groups.values() if max(r['sides'])==1],key=lambda r:(r['date'],r['a'],r['b'],r['map']))
    return rows,names,manifest,raw,quarantine

def load_rankings():
    rankings={};manifest=[]
    for provider in ['hltv','valve']:
        file=ROOT/f'data/rankings/{provider}_points.csv'
        manifest.append(dict(file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        rankings[provider]={}
        with file.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                name=key(row['team']);points=int(row['points'])
                if points<0:raise ValueError('积分必须非负')
                rankings[provider][name]=max(points,rankings[provider].get(name,0))
    medians={p:float(np.median(list(values.values()))) for p,values in rankings.items()}
    return rankings,medians,manifest
