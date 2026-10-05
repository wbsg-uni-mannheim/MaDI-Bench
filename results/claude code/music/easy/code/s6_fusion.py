import pandas as pd, numpy as np, json, sys, re
sys.path.insert(0,'work'); from common import key
a=pd.read_pickle('work/state/normalized.pkl').merge(pd.read_pickle('work/state/membership.pkl')[['id','cluster_id']],on='id')
PRI={'discogs':0,'musicbrainz':1,'lastfm':2}   # documented tie-break order (lastfm titles carry injected tags/dropped words)
a['pri']=a.source.map(PRI)
a=a.sort_values(['cluster_id','pri','id'])
def vote_text(m, col):
    v=m[m[col].notna()&(m[col].astype(str).str.strip()!='')]
    if v.empty: return None, 'missing'
    k=v[col].map(key)
    cnt=k.value_counts()
    top=cnt[cnt==cnt.max()].index
    cand=v[k.isin(top)].sort_values(['pri','id']).iloc[0]
    return cand[col], f'vote:{cand.source}' if cnt.max()>1 else f'priority:{cand.source}'
def pick_first(m,col,empty=lambda x: x is None or (isinstance(x,float) and np.isnan(x))):
    for _,r in m.iterrows():
        if not empty(r[col]): return r[col], r.source
    return None,'missing'
def fuse_date(m):
    v=m[m.date.notna()]
    if v.empty: return None,'missing'
    ds=v.date.tolist(); srcs=v.source.tolist()
    if len(set(ds))==1: return ds[0],'agree'
    # a YYYY-01-01 value is often a year-only placeholder: prefer a more specific date of the same year
    spec=[d for d in ds if not d.endswith('-01-01')]
    for d,s in zip(ds,srcs):
        if d.endswith('-01-01') and any(x[:4]==d[:4] for x in spec):
            best=[x for x in spec if x[:4]==d[:4]][0]; return best,'specific_over_placeholder'
    return ds[0],'priority:'+srcs[0]
def fuse_dur(m):
    v=m[m.duration.notna()]
    if v.empty: return None,'missing'
    vals=v.duration.astype(int).tolist()
    sup=[sum(abs(x-y)<=5 for y in vals) for x in vals]
    i=int(np.argmax(sup))  # first max -> source priority order
    return vals[i], ('agree' if sup[i]>1 else 'priority:'+v.source.iloc[i])
out=[];prov=[]
for cid,m in a.groupby('cluster_id',sort=True):
    r={'_id':cid,'id':m.id.iloc[0]}  # id: representative source-native id (priority order)
    r['name'],p1=vote_text(m,'name')
    art=m[m.artist.notna()]
    if len(art):
        # artist: token-based comparison target; discogs gives complete artist credit, lastfm abbreviates
        x=art.iloc[0]; av=x.artist
        if x.source=='musicbrainz' and av.count(',')==1:
            l,f=[t.strip() for t in av.split(',')]; av=('The '+l) if f.lower()=='the' else (f+' '+l).strip()
        r['artist'],p2=av,'priority:'+x.source
    else: r['artist'],p2=None,'missing'
    r['release-date'],p3=fuse_date(m)
    c=m[m.country.notna()]
    r['release-country'],p4=(c.country.iloc[0],'priority:'+c.source.iloc[0]) if len(c) else (None,'missing')
    L=[json.loads(x) for x in m.label]; L=[x for x in L if x]
    r['label']='|'.join(L[0]) if L else None
    gg=m.genre.dropna(); r['genre']=gg.iloc[0] if len(gg) else None
    T=[(json.loads(t),s) for t,s in zip(m.tracks,m.source)]; T=[x for x in T if x[0]]
    r['tracks']=json.dumps(T[0][0],ensure_ascii=False) if T else None; p7=('priority:'+T[0][1]) if T else 'missing'
    d,p8=fuse_dur(m); r['duration']=d
    out.append(r); prov.append(dict(_id=cid,name=p1,artist=p2,date=p3,country=p4,tracks=p7,duration=p8,members='|'.join(m.id)))
f=pd.DataFrame(out)
f['duration']=f.duration.astype('Int64')
cols=['_id','id','name','artist','release-date','release-country','label','genre','tracks','duration']
f[cols].to_csv('submission/fused.csv',index=False)
pd.DataFrame(prov).to_csv('work/state/fusion_provenance.csv',index=False)
print(f[cols].notna().mean())
