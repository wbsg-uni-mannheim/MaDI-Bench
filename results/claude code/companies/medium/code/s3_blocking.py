"""Stage 3: blocking. Union of (a) exact normalized-name key, (b) acronym key, (c) char-3gram TF-IDF
top-k nearest neighbours per record for every source pair (both directions), (d) within-fullcontact key
(fullcontact contains duplicate rows). Exports full cross-source candidate set."""
import pandas as pd, numpy as np, re, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
U=pd.read_pickle('work/state/s2_normalized.pkl')
SRC=['dbpedia','forbes','fullcontact']
pairs={}
def add(a,b,how):
    if a==b: return
    key=tuple(sorted((a,b))); pairs.setdefault(key,set()).add(how)
# (a) exact key
for k,g in U.groupby('k'):
    ids=g.record_id.tolist()
    if len(ids)>50: continue
    for i in range(len(ids)):
        for j in range(i+1,len(ids)): add(ids[i],ids[j],'key')
# (b) acronym: short all-caps-ish name vs initials of long name (stopwords dropped)
def initials(k):
    t=[w for w in k.split() if w not in {'and','of','the','de'}]
    return ''.join(w[0] for w in t) if len(t)>=3 else None
U['acr']=U.k.map(initials)
short=U[U.k.str.fullmatch(r'[a-z]{3,6}')]
acr=U[U.acr.notna()]
m=short.merge(acr,left_on='k',right_on='acr',suffixes=('','_2'))
for a,b in zip(m.record_id,m.record_id_2): add(a,b,'acronym')
# (c) tf-idf char n-gram kNN across source pairs
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,3),min_df=1,sublinear_tf=True)
X=vec.fit_transform(U.k.fillna(''))
K=8
for s in SRC:
    for t in SRC:
        if s==t: continue
        qi=np.where(U.source==s)[0]; ti=np.where(U.source==t)[0]
        nn=NearestNeighbors(n_neighbors=K,metric='cosine').fit(X[ti])
        d,idx=nn.kneighbors(X[qi])
        for r,q in enumerate(qi):
            for dd,j in zip(d[r],idx[r]):
                if 1-dd>=0.35: add(U.record_id.iat[q],U.record_id.iat[ti[j]],'tfidf')
# (d) within fullcontact near-duplicates
fi=np.where(U.source=='fullcontact')[0]
nn=NearestNeighbors(n_neighbors=5,metric='cosine').fit(X[fi]); d,idx=nn.kneighbors(X[fi])
for r,q in enumerate(fi):
    for dd,j in zip(d[r],idx[r]):
        if 1-dd>=0.6: add(U.record_id.iat[q],U.record_id.iat[fi[j]],'tfidf_within')
src=dict(zip(U.record_id,U.source))
C=pd.DataFrame([(a,b,src[a],src[b],'|'.join(sorted(h))) for (a,b),h in pairs.items()],columns=['id1','id2','s1','s2','how'])
C.to_pickle('work/state/s3_candidates.pkl')
cross=C[C.s1!=C.s2]
import os; os.makedirs('submission/blocking',exist_ok=True)
cross[['id1','id2']].to_csv('submission/blocking/candidates.csv',index=False)
n={s:(U.source==s).sum() for s in SRC}
full=n['dbpedia']*n['forbes']+n['dbpedia']*n['fullcontact']+n['forbes']*n['fullcontact']
diag=dict(cross_candidates=len(cross),within_candidates=int((C.s1==C.s2).sum()),full_cross=int(full),reduction_ratio=1-len(cross)/full,
  by_pair=cross.groupby(['s1','s2']).size().astype(int).to_dict(), by_how=C.how.value_counts().to_dict())
inv=set(cross.id1)|set(cross.id2)
diag['records_with_zero_cross_candidates']={s:int(((U.source==s)&~U.record_id.isin(inv)).sum()) for s in SRC}
diag={k:(str(v) if isinstance(v,dict) else v) for k,v in diag.items()}
print(json.dumps(diag,indent=1))
json.dump(diag,open('work/state/s3_diag.json','w'))
