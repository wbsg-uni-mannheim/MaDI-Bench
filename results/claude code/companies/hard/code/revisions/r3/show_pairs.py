import pandas as pd, sys
P=pd.read_pickle('state/pair_decisions.pkl'); R=pd.read_pickle('state/records.pkl')
P['nm1']=R.pname.values[P.i]; P['nm2']=R.pname.values[P.j]
def show(x, n=400):
    for r in x.head(n).itertuples():
        print(f"{r.s1[:3]}-{r.s2[:3]} {str(r.nm1)[:32]:32} | {str(r.nm2)[:32]:32} ns={r.ns:.2f} c={r.country} y={r.year} ci={r.city} in={r.industry} a={r.assets} r={r.revenue} p={r.people} e={r.emb:.2f} {'ACC' if r.accept else ''} {r.rule}")
