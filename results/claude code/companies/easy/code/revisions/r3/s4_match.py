"""Stage 4c: pair decisions = deterministic rules + documented manual overrides, then 1:1 (mutual best) per source pair
at fullcontact-duplicate-group level."""
import os, re, json, pandas as pd, numpy as np
from collections import Counter
W=os.path.dirname(os.path.abspath(__file__))
F=pd.read_csv(f"{W}/state/bands.csv"); N=pd.read_csv(f"{W}/state/normalized.csv").set_index("record_id")
GENERIC=set("industry industries international intl enterprises companies and of foods inds worldwide sys".split())
DIVISION=set("""weather animation studios studio media networks network telecom television tv radio sport sports records pictures
publishing outdoor interactive express city market planet mobile platforms solutions services systems construction property
properties energy electronics securities capital pharmaceuticals asset management airplanes renovables razorfish""".split())
GEO=set("""america americas american usa us north south east west asia asian pacific europe european africa india indian canada canadian
uk china chinese japan japanese australia australian germany deutschland france brasil brazil mexico singapore hong kong korea taiwan
ireland italia italy espana spain nederland netherlands international global worldwide middle emirates uae philippines malaysia
thailand indonesia vietnam russia poland turkey israel sweden norway denmark finland switzerland austria belgium new zealand nz
argentina chile colombia peru nigeria kenya egypt saudi arabia qatar pakistan bangladesh dubai abu dhabi beijing shanghai""".split())
df=Counter(t for k in N.name_key.fillna("") for t in set(k.split()))
STOP={"of","and","the","de","la","for","&"}
def keys(rid):
    r=N.loc[rid]; ks=set()
    if isinstance(r.name_key,str): ks.add(r.name_key)
    if isinstance(r.alt_key,str): ks|=set(r.alt_key.split("|"))
    return ks
def acronym(a,b):
    ta=a.split(); tb=[t for t in b.split() if t not in STOP]
    return len(ta)==1 and len(tb)>=3 and a=="".join(t[0] for t in tb)
def distinctive_shared(ka,kb):
    ta=set(" ".join(ka).split()); tb=set(" ".join(kb).split())
    return {t for t in ta&tb if t not in GEO and t not in STOP and t not in GENERIC and df.get(t,0)<=40 and len(t)>1}
man=pd.read_csv(f"{W}/manual_decisions.csv") if os.path.exists(f"{W}/manual_decisions.csv") else pd.DataFrame(columns=["g1","g2","decision","note"])
MAN={(r.g1,r.g2):r.decision for r in man.itertuples()}
rule=[];L=[]
for r in F.itertuples():
    ka,kb=keys(r.id1),keys(r.id2)
    if not ka or not kb: rule.append("no_name"); L.append(0); continue
    ds=distinctive_shared(ka,kb)
    extras=[set(x.split())^set(y.split()) for x in ka for y in kb if set(x.split())<=set(y.split()) or set(y.split())<=set(x.split())]
    ex=min(extras,key=len) if extras else None
    acro=any(acronym(x,y) or acronym(y,x) for x in ka for y in kb)
    m=MAN.get((r.g1,r.g2))
    if m is not None: w,l=("manual_accept",3) if m=="accept" else ("manual_reject",0)
    elif r.band=="auto": w,l="key_eq",3
    elif r.fin_exact and (ds or acro or r.key_eq): w,l="fin_exact+name",3
    elif r.key_eq: w,l="key_eq_conflict_review",0
    elif r.ratio>=0.92 and not r.geo_extra and r.con==0 and (r.country==1 or r.sup>0): w,l="near_exact",2
    elif ex is not None and ex and ex<=GENERIC and not r.paren and r.con==0 and r.country==1 and r.emb>=0.75: w,l="generic_suffix",2
    elif acro and r.con==0 and r.country==1: w,l="acronym",2
    elif (ex is not None and ex and not (ex&GEO) and not (ex&DIVISION) and not r.paren and r.emb>=0.75 and r.con==0
          and r.country==1 and (r.city==1 or r.year==1)): w,l="containment+city/year",1
    elif r.band=="review": w,l="review_unresolved",0
    else: w,l="reject",0
    rule.append(w); L.append(l)
F["rule"]=rule; F["level"]=L
F["score"]=F.level*10+F.sup-F.con+F.emb+F.ratio+3*F.fin_exact
# group-level edges: best member pair per (g1,g2)
E=F[F.level>0].sort_values("score",ascending=False).drop_duplicates(["g1","g2"]).copy()
E["sp"]=E.src1+"-"+E.src2
b1=E.groupby(["sp","g1"]).score.transform("max"); b2=E.groupby(["sp","g2"]).score.transform("max")
# dbpedia-forbes: strict mutual best. *-fullcontact: each fullcontact group keeps only its best partner per source,
# but a dbpedia/forbes record may absorb several fullcontact groups (fullcontact holds near-duplicate profiles).
E["mutual_best"]=np.where(E.sp=="dbpedia-forbes",(E.score==b1)&(E.score==b2),E.score==b2)
F=F.merge(E[["g1","g2","mutual_best"]],on=["g1","g2"],how="left")
F["accepted"]=(F.level>0)&(F.mutual_best==True)
F.to_csv(f"{W}/state/decisions.csv",index=False)
Gd=F.sort_values("level",ascending=False).drop_duplicates(["g1","g2"])
print(Gd.rule.value_counts().to_string())
print("accepted group edges",Gd.accepted.sum(), Gd[Gd.accepted].groupby(["src1","src2"]).size().to_dict())
print("level>0 not mutual best:",((Gd.level>0)&~Gd.accepted).sum())
