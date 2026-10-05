import pandas as pd, numpy as np, json, sys
from rapidfuzz import fuzz, process
a=pd.read_pickle('work/state/normalized.pkl')
c=pd.read_pickle('work/state/features.pkl')
CFG=dict(gate_name=0.75, gate_artist=0.4, id_min=0.80, accept=0.85, alt_id_min=0.65, alt_ft=0.8, alt_ntr=3, alt_nset=0.9, noev_id_min=0.9)
json.dump(CFG,open('work/state/match_cfg.json','w'))
# plausibility gate before expensive fuzzy-track comparison
g=c[(c.nset>=CFG['gate_name'])&((c['as']>=CFG['gate_artist'])|c['as'].isna())].copy()
tks=[[t for t in json.loads(s)] for s in a.tracks_key]
def ftrack(i,j):
    A,B=tks[i],tks[j]
    if not A or not B: return np.nan
    if len(A)>len(B): A,B=B,A
    Bs=list(B); m=0
    for t in A:
        r=process.extractOne(t,Bs,scorer=fuzz.token_sort_ratio,score_cutoff=80)
        if r: m+=1; Bs.pop(r[2])
        if not Bs: break
    return m/len(A)
g['ft']=[ftrack(i,j) for i,j in zip(g.i1,g.i2)]
nsrc=a.source.values
g['s1']=nsrc[g.i1]; g['s2']=nsrc[g.i2]
ntr=np.array([len(t) for t in tks])
g['nt_min']=[min(ntr[i],ntr[j]) for i,j in zip(g.i1,g.i2)]
g['nt_ratio']=[min(ntr[i],ntr[j])/max(ntr[i],ntr[j]) if ntr[i] and ntr[j] else np.nan for i,j in zip(g.i1,g.i2)]
name=0.5*g.ns+0.5*g.nset
art=g['as'].fillna(0.6)
g['id_score']=0.5*name+0.5*art
def term(x,f): return x.map(f).fillna(0)
# issue-level evidence: tracks, duration, date, country (missing -> 0, i.e. unknown)
g['t_tr']=((g.ft-0.5)*2).fillna(0)
g['t_du']=g.dd.map(lambda d: np.nan if pd.isna(d) else (1 if d<=5 else 0.3 if d<=60 else -0.5)).fillna(0)
g['t_da']=np.where(g.de==1,1,np.where(g.dy==1,0.3,np.where(g.dy==0,-1,0)))
g['t_co']=np.where(g.ce==1,1,np.where(g.ce==0,-0.5,0))
g['score']=g.id_score+0.25*g.t_tr+0.08*g.t_du+0.12*g.t_da+0.06*g.t_co
g.to_pickle('work/state/scored.pkl')
print(len(g)); print(g.score.describe())
