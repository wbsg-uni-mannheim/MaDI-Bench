"""Writes a human-review listing of records grouped by spec block with current cluster labels."""
import pandas as pd, sys
a = pd.read_csv('work/state/s1_translated.csv', dtype=str, keep_default_na=False)
n = pd.read_csv('work/state/s2_features.csv', dtype=str, keep_default_na=False)
m = pd.read_csv('work/state/s5_membership.csv', dtype=str, keep_default_na=False)
d = a.merge(n,on=['id','source']).merge(m[['record_id','cluster_id']], left_on='id', right_on='record_id')
d['spec'] = d.apply(lambda r: (r.chip or '?') + ('/'+str(r.vram) if r.vram else '') if r.ptype=='GPU' else (r.cap_gb or '?'), axis=1)
d['blk'] = d.ptype+'|'+d.brand_k+'|'+d.spec
d['sid'] = d.id.str.replace('products_','')
cl = {c:i for i,c in enumerate(sorted(d.cluster_id.unique()))}
d['cl'] = d.cluster_id.map(cl)
d['csize'] = d.groupby('cluster_id').id.transform('size')
out=[]
for blk, g in sorted(d.groupby('blk'), key=lambda x: x[0]):
    out.append(f'### {blk} ({len(g)})')
    for r in g.sort_values(['cl','title']).itertuples():
        mark = f'C{r.cl}' if r.csize>1 else '  -'
        out.append(f'{mark:>6} {r.sid:<12} {r.title[:120]} || {r.model_number[:30]}')
open('work/state/review.txt','w').write('\n'.join(out))
print(len(out))
