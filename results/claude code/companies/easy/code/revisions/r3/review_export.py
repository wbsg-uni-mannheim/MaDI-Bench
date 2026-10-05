"""Exports unresolved pairs (group level) for pair-by-pair decisions -> state/review.txt, state/review_pairs.csv"""
import pandas as pd
F=pd.read_csv("state/decisions.csv"); N=pd.read_csv("state/normalized.csv").set_index("record_id")
R=F[F.rule.isin(["review_unresolved","key_eq_conflict_review"])].sort_values("score",ascending=False).drop_duplicates(["g1","g2"])
R=R.sort_values(["src1","src2","name1"]).reset_index(drop=True)
C={"United States of America":"USA","United States":"US","United Kingdom of Great Britain and Northern Ireland":"UK*","United Kingdom":"UK"}
def d(rid):
    r=N.loc[rid]; f=lambda v:"" if pd.isna(v) else str(v)
    p=[f(r.name_raw), C.get(f(r.country_raw),f(r.country_raw)), f(r.city)[:18], f(r.founded)[:4]]
    if r.source!="fullcontact": p+=[f(r.industry_raw)[:18], "r="+f(r.revenue_raw)+" a="+f(r.assets_raw)]
    return "/".join(p)
R[["g1","g2","src1","src2","rule"]].to_csv("state/review_pairs.csv",index_label="rid")
with open("state/review.txt","w") as o:
    for i,r in R.iterrows():
        o.write(f"{i} {d(r.id1)} <=> {d(r.id2)} [fx{r.fin_exact} c{r.country} ci{r.city} y{r.year} f{r.fin} e{r.emb:.2f}]\n")
print(len(R))
