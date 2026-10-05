"""Stage 6: fusion. Attribute-wise resolution from the final membership's own member records.
Source preference (documented heuristic, not learned): numeric financials forbes>dbpedia; industry forbes>dbpedia;
names forbes>dbpedia>fullcontact; founded/city/keypeople by normalized vote with dbpedia tie-break;
country by weighted vote (fullcontact weight 0.5 because its country field is observably noisy)."""
import pandas as pd, json, re
from collections import Counter
U=pd.read_pickle('work/state/s2_normalized.pkl')
m=pd.read_pickle('work/state/s5_membership.pkl')
U=U.merge(m[['record_id','cluster_id']],on='record_id')
PRI={'forbes':0,'dbpedia':1,'fullcontact':2}
U['_p']=U.source.map(PRI)
U=U.sort_values(['cluster_id','_p','record_id'])
def out_name(g):
    for r in g.itertuples():
        if isinstance(r.name,str) and r.name:
            n=re.sub(r'\s*\([^)]*\)\s*$','',r.name).strip() or r.name
            return n,r.record_id
    return g.name_raw.iloc[0],g.record_id.iloc[0]
def vote(g,col,weights=None,key=lambda v:v):
    c=Counter(); first={}
    for r in g.itertuples():
        v=getattr(r,col)
        if v is None or (not isinstance(v,(list,str)) and pd.isna(v)): continue
        k=key(v); c[k]+=(weights or {}).get(r.source,1.0); first.setdefault(k,(v,r.record_id))
    if not c: return None,None,0
    best=max(c.values()); ks=[k for k in c if c[k]==best]
    k=ks[0]   # insertion order = source priority order -> deterministic tie-break
    return first[k][0],first[k][1],len(c)
def pref(g,col,order):
    for s in order:
        for r in g[g.source==s].itertuples():
            v=getattr(r,col)
            if v is not None and not (not isinstance(v,(list,str)) and pd.isna(v)): return v,r.record_id
    return None,None
rows=[];prov=[]
for cid,g in U.groupby('cluster_id',sort=True):
    rec={'_id':cid,'id':cid}; pv={'_id':cid}
    rec['name'],pv['name']=out_name(g)
    y,pv['founded'],nf=vote(g,'founded_year',key=int)
    rec['founded']=f'{int(y):04d}-01-01' if y is not None else None
    rec['country'],pv['country'],nc=vote(g,'country',{'forbes':1.0,'dbpedia':1.0,'fullcontact':0.5})
    rec['city'],pv['city'],_=vote(g,'city',{'dbpedia':1.0,'fullcontact':1.0},key=lambda v:v.lower())
    rec['industry'],pv['industry']=pref(g,'industry',['forbes','dbpedia'])
    for a in ['assets','revenue']:
        v,pv[a]=pref(g,a,['forbes','dbpedia'])
        rec[a]=int(v) if v is not None else None
    kp,pv['keypeople'],_=vote(g,'keypeople',{'dbpedia':1.0,'fullcontact':1.0},key=lambda v:tuple(sorted(x.lower() for x in v)))
    rec['keypeople']=json.dumps(kp,ensure_ascii=False) if kp else None
    pv['n_members']=len(g); pv['conflicts']={'founded':nf>1,'country':nc>1}
    rows.append(rec); prov.append(pv)
F=pd.DataFrame(rows)
cols=['_id','id','name','founded','country','city','industry','assets','revenue','keypeople']
F=F[cols]
for a in ['assets','revenue']: F[a]=F[a].astype('Int64')
F.to_pickle('work/state/s6_fused.pkl')
json.dump(prov,open('work/state/s6_provenance.json','w'),ensure_ascii=False)
print(F.notna().mean().round(3).to_dict())
