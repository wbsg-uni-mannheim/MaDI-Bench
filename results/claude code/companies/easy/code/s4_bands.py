"""Stage 4b: evidence bands. Collapses exact-duplicate fullcontact rows into groups; assigns auto_accept / review / reject."""
import os, re, pandas as pd, numpy as np
W=os.path.dirname(os.path.abspath(__file__))
F=pd.read_csv(f"{W}/state/features.csv"); N=pd.read_csv(f"{W}/state/normalized.csv")
fc=N[N.source=="fullcontact"].copy()
fc["grp"]=fc.groupby(fc.name_raw.fillna(fc.record_id)).record_id.transform("first")
G=dict(zip(fc.record_id,fc.grp)); pd.Series(G).rename("group").rename_axis("record_id").to_csv(f"{W}/state/fc_groups.csv")
NI=N.set_index("record_id")
F["g1"]=F.id1.map(lambda x:G.get(x,x)); F["g2"]=F.id2.map(lambda x:G.get(x,x))
F["sup"]=(F.country==1).astype(int)+(F.city==1)+(F.year==1)+(F.fin==1)+F.fin_exact
F["con"]=(F.country==-1).astype(int)+(F.city==-1)+(F.year==-1)+((F.fin==-1)&(F.fin_exact==0))
OKPAREN=r"company|companies|corporation|business|enterprise|conglomerate|firm|holding company|brand|retailer|group|bank"
def paren(rid):
    m=re.search(r"\(([^)]*)\)\s*$",str(NI.loc[rid,"name_raw"])) if rid.startswith("http://dbpedia") else None
    return bool(m) and not re.fullmatch(OKPAREN,m.group(1).strip().lower())
F["paren"]=[paren(a) or paren(b) for a,b in zip(F.id1,F.id2)]
auto=(F.key_eq==1)&~F.paren&((F.con==0)|((F.sup>=1)&(F.con<=1)))
review=((F.key_eq==1)&~auto) | ((F.key_eq==0)&((F.ratio>=0.85)|(F.tset>=0.99)|(F.fin_exact==1)|(F.emb>=0.85))
        &((F.country!=-1)|(F.fin_exact==1)|(F.emb>=0.85))&(F.con<=1))
F["band"]=np.where(auto,"auto",np.where(review,"review","reject"))
# group-level: one row per (g1,g2) = best-evidence member pair
F["rank"]=F.sup-F.con+F.emb
F=F.sort_values(["g1","g2","rank"],ascending=[True,True,False])
F.to_csv(f"{W}/state/bands.csv",index=False)
Gp=F.drop_duplicates(["g1","g2"])
print(Gp.band.value_counts().to_dict())
