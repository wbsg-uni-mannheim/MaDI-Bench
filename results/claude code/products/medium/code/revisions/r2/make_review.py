import pandas as pd, sys
n = pd.read_pickle('work/state/normalized.pkl').set_index('id')
cl = pd.read_csv('work/state/clusters.csv').set_index('id').cluster
n['cl'] = cl
n = n[n.n_product_type.isin(['GPU','SSD','HDD','USB_STICK']) | n.n_real_brand]
rep = n.groupby('cl').title.min(); n['rep'] = n.cl.map(rep)
n = n.sort_values(['n_product_type','n_brand','n_cap_gb','n_chip','rep','cl'], na_position='last')
out = []; last = None
for i, r in n.iterrows():
    key = (r.n_product_type, r.n_brand)
    if key != last: out.append(f'##### {key}'); last = key
    cap = r.n_cap_gb if r.n_cap_gb == r.n_cap_gb else (r.n_vram_gb if r.n_vram_gb == r.n_vram_gb else '')
    out.append(f'{r.cl[11:]:12s} {i[9:]:11s} {str(cap):7s} {str(r.n_chip or "")[:14]:14s} | {r.title[:95]} | {r.model_number[:22]}')
open('/tmp/review.txt','w').write('\n'.join(out)); print(len(out))
