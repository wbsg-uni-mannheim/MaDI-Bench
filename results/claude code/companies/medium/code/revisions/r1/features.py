import pandas as pd, numpy as np
from rapidfuzz import fuzz
def build(C,U):
    R=U.set_index('record_id')
    a=R.loc[C.id1].reset_index(drop=True); b=R.loc[C.id2].reset_index(drop=True)
    F=C.reset_index(drop=True).copy()
    F['n1']=a.name_raw.values; F['n2']=b.name_raw.values
    ka=a.k.values; kb=b.k.values
    F['k1']=ka; F['k2']=kb
    F['tsr']=[fuzz.token_sort_ratio(x,y) for x,y in zip(ka,kb)]
    F['ratio']=[fuzz.ratio(x.replace(' ',''),y.replace(' ','')) for x,y in zip(ka,kb)]
    F['extra1']=[' '.join(sorted(set(x.split())-set(y.split()))) for x,y in zip(ka,kb)]
    F['extra2']=[' '.join(sorted(set(y.split())-set(x.split()))) for x,y in zip(ka,kb)]
    def cmp(x,y):
        if pd.isna(x) or pd.isna(y): return 0
        return 1 if x==y else -1
    F['country']=[cmp(x,y) for x,y in zip(a.country,b.country)]
    F['c1']=a.country.values; F['c2']=b.country.values
    F['city']=[cmp(str(x).lower() if pd.notna(x) else x, str(y).lower() if pd.notna(y) else y) for x,y in zip(a.city,b.city)]
    F['yr']=[0 if (pd.isna(x) or pd.isna(y)) else (1 if abs(x-y)<=1 else -1) for x,y in zip(a.founded_year,b.founded_year)]
    def num(x,y):
        if pd.isna(x) or pd.isna(y) or x==0 or y==0: return 0
        r=max(x,y)/min(x,y); return 1 if r<=1.3 else (-1 if r>3 else 0)
    F['rev']=[num(x,y) for x,y in zip(a.revenue,b.revenue)]
    F['ast']=[num(x,y) for x,y in zip(a.assets,b.assets)]
    F['ind']=[cmp(x,y) for x,y in zip(a.industry,b.industry)]
    return F
