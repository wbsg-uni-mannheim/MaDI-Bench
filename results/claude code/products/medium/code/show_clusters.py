import pandas as pd, sys
n = pd.read_pickle('work/state/normalized.pkl').set_index('id')
cl = pd.read_csv('work/state/clusters.csv')
sz = cl.groupby('cluster').size()
minsz = int(sys.argv[1]) if len(sys.argv) > 1 else 5
maxsz = int(sys.argv[2]) if len(sys.argv) > 2 else 999
filt = sys.argv[3] if len(sys.argv) > 3 else None
for c in sz[(sz >= minsz) & (sz <= maxsz)].index:
    m = cl[cl.cluster == c].id
    rows = n.loc[m]
    if filt and not rows.n_product_type.fillna('').str.contains(filt).any(): continue
    print('====', c, len(m), dict(rows.source.value_counts()))
    for i, r in rows.sort_values('title').iterrows():
        print(f'  {i:22s} {str(r.n_product_type)[:4]:4s} {r.n_cap_gb if r.n_cap_gb==r.n_cap_gb else "":>7} {r.n_vram_gb if r.n_vram_gb==r.n_vram_gb else "":>4} | {r.title[:95]} | {r.model_number[:20]}')
