import pandas as pd, numpy as np, json, sys
from rapidfuzz import fuzz
a=pd.read_pickle('work/state/normalized.pkl')
c=pd.read_pickle('work/state/candidates.pkl')
nk=a.name_key.tolist(); ak=[json.loads(s) for s in a.artist_keys]
tk=[set(json.loads(s)) for s in a.tracks_key]
dur=a.duration.astype('float').values; date=a.date.tolist(); ctry=a.country.tolist(); src=a.source.tolist()
def abbrev(x,y,directed=False):
    """token alignment allowing initials/abbreviations (lastfm 'R. Trent', 'Bible o. the Devil').
    directed=True: x is the (possibly abbreviated/truncated) lastfm side; all its tokens must align."""
    tx,ty=x.split(),y.split()
    if not tx or not ty: return 0.0
    if not directed and len(tx)>len(ty): tx,ty=ty,tx
    used=set(); m=0; full=0
    for t in tx:
        for j,u in enumerate(ty):
            if j in used: continue
            if t==u or (len(t)==1 and u.startswith(t)) or (len(t)>=3 and u.startswith(t)):
                used.add(j); m+=1; full+= (t==u and len(t)>1); break
    if full==0: return 0.0
    return m/len(tx) * (0.9 if len(tx)<len(ty) else 1.0)
def asim(i,j):
    A,B=ak[i],ak[j]
    lf=None
    if src[i]=='lastfm': lf=0
    elif src[j]=='lastfm': lf=1
    if not A or not B or not any(A) or not any(B): return np.nan
    best=0
    for x in A:
        for y in B:
            if not x or not y: continue
            if lf is None: ab=abbrev(x,y)
            elif lf==0: ab=abbrev(x,y,True)
            else: ab=abbrev(y,x,True)
            best=max(best,fuzz.token_sort_ratio(x,y)/100,ab)
            xs,ys=x.replace(' ',''),y.replace(' ','')
            best=max(best,fuzz.ratio(xs,ys)/100)
            if min(len(xs),len(ys))>=3 and (xs.startswith(ys) or ys.startswith(xs)): best=max(best,0.85)
    # discogs multi-artist vs single combined string
    if len(A)>1 or len(B)>1:
        best=max(best,fuzz.token_set_ratio(' '.join(A),' '.join(B))/100*0.95)
    return best
rows=[]
for i,j in zip(c.i1.values,c.i2.values):
    x,y=nk[i],nk[j]
    ns=fuzz.token_sort_ratio(x,y)/100; nset=fuzz.token_set_ratio(x,y)/100
    ta,tb=tk[i],tk[j]
    if ta and tb:
        inter=len(ta&tb); tj=inter/len(ta|tb); tc=inter/min(len(ta),len(tb))
    else: tj=tc=np.nan
    dd=abs(dur[i]-dur[j]) if not (np.isnan(dur[i]) or np.isnan(dur[j])) else np.nan
    de=np.nan; ce=np.nan; dy=np.nan
    if date[i] and date[j]: de=float(date[i]==date[j]); dy=float(date[i][:4]==date[j][:4])
    if ctry[i] and ctry[j]: ce=float(ctry[i]==ctry[j])
    rows.append((ns,nset,asim(i,j),tj,tc,dd,de,dy,ce))
f=pd.DataFrame(rows,columns=['ns','nset','as','tj','tc','dd','de','dy','ce'])
c=pd.concat([c.reset_index(drop=True),f],axis=1)
c.to_pickle('work/state/features.pkl')
print(c.describe().T)
