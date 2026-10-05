"""Stage 3: blocking = union of (a) exact key, (b) char-3gram TF-IDF kNN, (c) cached-embedding kNN, per source pair."""
import os, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
W=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(W)
N=pd.read_csv(f"{W}/state/normalized.csv")
ids=pd.read_csv(f"{W}/state/emb_ids.csv").record_id
assert (ids.values==N.record_id.values).all()
E=np.load(f"{W}/state/emb_name.npy")
def keys(r):
    ks={r.name_key} if isinstance(r.name_key,str) and r.name_key else set()
    if isinstance(r.alt_key,str): ks|=set(r.alt_key.split("|"))
    return ks
N["keys"]=N.apply(keys,axis=1)
SRC=["dbpedia","forbes","fullcontact"]; idx={s:np.where(N.source==s)[0] for s in SRC}
pairs={}  # (i,j) -> set(methods)
def add(i,j,m):
    a,b=(i,j) if i<j else (j,i); pairs.setdefault((a,b),set()).add(m)
# (a) exact keys
inv={}
for i,ks in enumerate(N["keys"]):
    for k in ks: inv.setdefault(k,[]).append(i)
big=0
for k,l in inv.items():
    if len(l)>50: big+=1; continue
    for x in l:
        for y in l:
            if x<y and N.source[x]!=N.source[y]: add(x,y,"key")
# (b) tfidf char ngrams on name_key + alt keys
txt=N["keys"].map(lambda s:" ".join(sorted(s)))
V=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,3),min_df=1,sublinear_tf=True).fit(txt)
X=V.transform(txt)
K_T,TH_T=5,0.5; K_E,TH_E=5,0.6
for a in SRC:
    for b in SRC:
        if a>=b: continue
        ia,ib=idx[a],idx[b]
        S=(X[ia]@X[ib].T).toarray()
        for M,K,TH,name in [(S,K_T,TH_T,"tfidf"),(E[ia]@E[ib].T,K_E,TH_E,"emb")]:
            for r in range(len(ia)):
                top=np.argpartition(-M[r],K)[:K]
                for c in top:
                    if M[r,c]>=TH: add(ia[r],ib[c],name)
            for c in range(len(ib)):
                top=np.argpartition(-M[:,c],K)[:K]
                for r in top:
                    if M[r,c]>=TH: add(ia[r],ib[c],name)
rows=[(N.record_id[i],N.record_id[j],N.source[i],N.source[j],"+".join(sorted(m))) for (i,j),m in pairs.items()]
C=pd.DataFrame(rows,columns=["id1","id2","src1","src2","methods"])
C.to_csv(f"{W}/state/candidates.csv",index=False)
os.makedirs(f"{ROOT}/submission/blocking",exist_ok=True)
C[["id1","id2"]].to_csv(f"{ROOT}/submission/blocking/candidates.csv",index=False)
n={s:len(idx[s]) for s in SRC}; total=n["dbpedia"]*n["forbes"]+n["dbpedia"]*n["fullcontact"]+n["forbes"]*n["fullcontact"]
per=pd.concat([C.id1,C.id2]).value_counts()
diag=dict(stage="blocking",candidates=len(C),cross_pairs_total=total,reduction_ratio=1-len(C)/total,skipped_big_key_blocks=big,
  by_srcpair=C.groupby(["src1","src2"]).size().rename(lambda t:f"{t[0]}-{t[1]}" if isinstance(t,tuple) else t).to_dict() if False else {f"{a}-{b}":int(v) for (a,b),v in C.groupby(["src1","src2"]).size().items()},
  by_method={k:int(v) for k,v in C.methods.value_counts().items()},
  zero_candidate_records={s:int((~N.record_id[idx[s]].isin(per.index)).sum()) for s in SRC},
  max_cand_per_record=int(per.max()))
print(json.dumps(diag,indent=1))
json.dump(diag,open(f"{W}/state/blocking_diag.json","w"),indent=1)
