"""Stage 6: attribute-wise fusion from each cluster's own members, with per-cell provenance."""
import os, re, json, pandas as pd, numpy as np
from collections import Counter
W=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(W)
N=pd.read_csv(f"{W}/state/normalized.csv"); M=pd.read_csv(f"{W}/state/membership.csv")
N=N.merge(M[["record_id","cluster_id"]],on="record_id")
def clean_city(c):
    if not isinstance(c,str) or not c.strip(): return None
    c=re.sub(r"(?<=[a-z])(?=\d)","|",c)                  # 'New York1211 Avenue' -> split before digits
    c=re.sub(r"(?<=[a-z\.\)])(?=[A-Z])","|",c)           # 'GuangdongShenzhen' -> 'Guangdong|Shenzhen'
    segs=[s.strip(" ,") for s in c.split("|") if s.strip(" ,")]
    seg=segs[0] if ("," in segs[0] or len(segs)==1 or re.search(r"\d",segs[-1])) else segs[-1]
    seg=seg.split(",")[0].strip()
    seg=re.sub(r"\s*\(.*?\)","",seg).strip()
    return seg or None
LIMIT={"assets":4e12,"revenue":1e12}
PRI={"name":["forbes","dbpedia","fullcontact"],"country":["forbes","dbpedia","fullcontact"],"industry":["forbes","dbpedia"],
     "founded":["dbpedia","fullcontact"],"city":["fullcontact","dbpedia"],"assets":["forbes","dbpedia"],"revenue":["forbes","dbpedia"]}
N["city_c"]=N.city.map(clean_city)
rows=[];prov=[]
for cid,g in N.groupby("cluster_id",sort=True):
    o={"_id":cid,"id":cid}
    for a,order in PRI.items():
        col="city_c" if a=="city" else a
        vals=[(s,v) for s in order for v in g[g.source==s][col] if isinstance(v,str) and v.strip() or (isinstance(v,(int,float)) and not pd.isna(v))]
        if a in LIMIT: vals=[(s,v) for s,v in vals if 0<=float(v)<=LIMIT[a]]   # schema range; out-of-range treated as unusable
        if not vals: o[a]=None; continue
        if a in ("country","industry","founded"):
            key=(lambda v:v[:4]) if a=="founded" else (lambda v:v)
            cnt=Counter(key(v) for _,v in vals); top=max(cnt.values())
            # majority vote; ties broken by source priority order
            chosen=next(v for s,v in vals if cnt[key(v)]==top); rule="vote" if len(cnt)>1 else "single"
        else:
            chosen=vals[0][1]; rule="priority"
        o[a]=chosen
        prov.append(dict(cluster_id=cid,attribute=a,value=chosen,rule=rule,n_candidates=len(vals),n_distinct=len(set(map(str,[v for _,v in vals]))),
                         sources="|".join(sorted(set(s for s,_ in vals)))))
    kp=[]
    for v in g.keypeople.dropna():
        for p in json.loads(v):
            if p.lower() not in [x.lower() for x in kp]: kp.append(p)
    o["keypeople"]=json.dumps(kp[:20],ensure_ascii=False) if kp else None
    rows.append(o)
F=pd.DataFrame(rows)
for c in ["assets","revenue"]: F[c]=F[c].map(lambda v: None if v is None or pd.isna(v) else str(int(round(float(v)))))
cols=["_id","id","name","founded","country","city","industry","assets","revenue","keypeople"]
F=F[cols]
os.makedirs(f"{ROOT}/submission",exist_ok=True)
F.to_csv(f"{ROOT}/submission/fused.csv",index=False)
pd.DataFrame(prov).to_csv(f"{W}/state/fusion_provenance.csv",index=False)
M.to_csv(f"{ROOT}/submission/membership.csv",index=False)
C=pd.read_csv(f"{W}/state/correspondences.csv"); C[["id1","id2","score"]].to_csv(f"{ROOT}/submission/correspondences.csv",index=False)
print(F.notna().mean().round(3).to_dict())
