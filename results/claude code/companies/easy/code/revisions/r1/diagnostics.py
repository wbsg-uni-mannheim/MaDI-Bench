"""Label-free diagnostic panel computed from the saved submission artifacts. Appends to diagnostics.jsonl."""
import os, json, datetime, pandas as pd, numpy as np
W=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(W); S=f"{ROOT}/submission"
F=pd.read_csv(f"{S}/fused.csv",dtype=str); M=pd.read_csv(f"{S}/membership.csv",dtype=str)
C=pd.read_csv(f"{S}/correspondences.csv",dtype=str); B=pd.read_csv(f"{S}/blocking/candidates.csv",dtype=str)
src={}
for s in ["dbpedia","forbes","fullcontact"]:
    for i in pd.read_csv(f"{ROOT}/task/input/data/{s}.csv",dtype=str).id: src[i]=s
schema=json.load(open(f"{ROOT}/task/input/schemamatching/target_schema.json"))
cldr=set(pd.read_csv(f"{ROOT}/task/input/schemamatching/CLDR_Country_Taxonomy.csv")["Country Name"])
gics=set(pd.read_csv(f"{ROOT}/task/input/schemamatching/GICS_Industry_Taxonomy.csv")["Industry Name"])
d={}
d["records_in_sources"]=len(src); d["membership_rows"]=len(M); d["membership_unique_ids"]=M.record_id.nunique()
d["unresolved_membership_ids"]=int((~M.record_id.isin(src)).sum()); d["source_mismatch"]=int((M.source!=M.record_id.map(src)).sum())
d["coverage_by_source"]={s:round(M[M.source==s].record_id.nunique()/sum(1 for v in src.values() if v==s),4) for s in ["dbpedia","forbes","fullcontact"]}
d["fused_rows"]=len(F); d["fused_ids_eq_clusters"]=set(F._id)==set(M.cluster_id); d["fused_id_unique"]=F._id.is_unique
bk=set(map(frozenset,zip(B.id1,B.id2)))
d["candidates"]=len(B); d["correspondences"]=len(C)
d["corr_in_candidates"]=round(np.mean([frozenset(p) in bk for p in zip(C.id1,C.id2)]),4)
cl=dict(zip(M.record_id,M.cluster_id))
d["corr_same_cluster"]=round(np.mean([cl[a]==cl[b] for a,b in zip(C.id1,C.id2)]),4)
d["corr_cross_source"]=round(np.mean([src[a]!=src[b] for a,b in zip(C.id1,C.id2)]),4)
sz=M.groupby("cluster_id").size(); ns=M.groupby("cluster_id").source.nunique()
d["cluster_size_dist"]={int(k):int(v) for k,v in sz.value_counts().sort_index().items()}
d["sources_per_cluster"]={int(k):int(v) for k,v in ns.value_counts().sort_index().items()}
d["singleton_share"]=round((sz==1).mean(),4)
mult=M.groupby(["cluster_id","source"]).size(); d["clusters_with_2plus_same_source"]={s:int((mult.xs(s,level=1)>1).sum()) for s in ["dbpedia","forbes","fullcontact"] if s in mult.index.get_level_values(1)}
d["corr_by_srcpair"]=pd.Series(["-".join(sorted([src[a],src[b]])) for a,b in zip(C.id1,C.id2)]).value_counts().to_dict()
dens={c:round(F[c].fillna("").str.strip().ne("").mean(),4) for c in F.columns}; d["fused_density"]=dens
d["country_valid"]=round(F.country.dropna().isin(cldr).mean(),4); d["industry_valid"]=round(F.industry.dropna().isin(gics).mean(),4)
d["founded_valid"]=round(F.founded.dropna().str.match(r"^\d{4}-\d{2}-\d{2}$").mean(),4)
yr=F.founded.dropna().str[:4].astype(int); d["founded_in_range"]=round(((yr>=1700)&(yr<=2016)).mean(),4)
for c,mx in [("assets",4e12),("revenue",1e12)]:
    v=F[c].dropna().astype(float); d[f"{c}_in_range"]=round(((v>=0)&(v<=mx)).mean(),4)
d["name_len_ok"]=round(F.name.dropna().str.len().between(1,200).mean(),4)
kp=F.keypeople.dropna().map(json.loads); d["keypeople_list_ok"]=round(kp.map(lambda l:1<=len(l)<=20).mean(),4) if len(kp) else None
d["timestamp"]=datetime.datetime.utcnow().isoformat(); d["inputs"]="submission/*.csv"
d["revision"]=open(f"{W}/REVISION").read().strip() if os.path.exists(f"{W}/REVISION") else "unversioned"
with open(f"{W}/diagnostics.jsonl","a") as o: o.write(json.dumps(d)+"\n")
print(json.dumps(d,indent=1))
