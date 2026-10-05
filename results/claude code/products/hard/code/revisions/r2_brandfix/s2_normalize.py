"""Stage 2: normalization of identity features used for blocking/matching.
Raw values stay in s1_translated.csv; this writes derived columns only."""
import pandas as pd, json, re, sys
sys.path.insert(0,'work')
from normlib import *
a = pd.read_csv('work/state/s1_translated.csv', dtype=str, keep_default_na=False)
rows=[]
for _,r in a.iterrows():
    title, desc = r.title, r.description
    tb = brands_in_text(title)
    tb_hi = [b for b in tb if b not in LOWPRI]
    fb, how = brand_from_field(r.brand)
    # brand key: title partner brand first (titles are clean), then field, then description
    if tb_hi: bk = tb_hi[0]; bsrc='title'
    elif fb: bk = fb; bsrc='field_'+how
    elif tb: bk = tb[0]; bsrc='title_lowpri'
    elif brand_from_lines(title+' '+r.model+' '+r.model_number):
        bk = brand_from_lines(title+' '+r.model+' '+r.model_number); bsrc='line_title'
    else:
        db = [b for b in brands_in_text(desc+' '+re.sub(r'https?://[^/]+','',r.url)) if b not in LOWPRI]
        lb = brand_from_lines(desc)
        bk = db[0] if db else (lb or ''); bsrc = 'desc' if db else ('line_desc' if lb else 'none')
    # if field brand exact and title-brand differs (e.g. title 'HP' but field 'Seagate'), keep title but note
    pf = ptype_from_field(r.product_type)
    pt = ptype_from_text(title) or ptype_from_text(r.model)
    pdsc = ptype_from_text(desc)
    if True:
        ptype = pf or pt or pdsc or ''; psrc = 'field' if pf else ('title' if pt else ('desc' if pdsc else 'none'))
    caps_t = capacities(title)
    cap=None; csrc='none'
    if ptype in ('SSD','HDD','USB_STICK',''):
        c = [x for x in caps_t if x[1]>=1]
        if c: cap = c[0][1]; csrc='title'
        else:
            v = parse_num(r.storage_gb)
            if v:
                cap = v*1000 if (ptype=='HDD' and v<=24) else v; csrc='field'
            else:
                cd = [x for x in capacities(desc) if x[1]>=1]
                if cd: cap = cd[0][1]; csrc='desc'
    chip = vram = None
    if ptype=='GPU':
        chip = gpu_chip(title) or gpu_chip(r.model) or gpu_chip(r.chipset_name) or gpu_chip(desc)
        vram = vram_from(title) or vram_from(r.model_number) or vram_from(desc)
        if vram is None:
            v = parse_num(r.vram_gb)
            if v: vram = v/1024 if v>=512 else v
    cds = codes(title, r.model_number, r.model)
    rows.append(dict(id=r.id, source=r.source, brand_k=bk, brand_src=bsrc, ptype=ptype, ptype_src=psrc,
                     cap_gb=cap, cap_src=csrc, chip=chip or '', vram=vram, codes='|'.join(sorted(cds))))
n = pd.DataFrame(rows)
n.to_csv('work/state/s2_features.csv', index=False)
print(n.brand_src.value_counts(), n.ptype.value_counts(), n.ptype_src.value_counts(), n.cap_src.value_counts(), sep='\n')
print('gpu no chip', ((n.ptype=='GPU')&(n.chip=='')).sum(), 'gpu no vram', ((n.ptype=='GPU')&n.vram.isna()).sum())
print('no brand', (n.brand_k=='').sum(), 'no codes', (n.codes=='').sum())
