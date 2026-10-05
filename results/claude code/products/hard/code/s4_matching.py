"""Stage 4: pairwise entity matching over blocking candidates (unsupervised, rule-based).
Evidence: shared model codes, IDF-weighted descriptive-token overlap, and hard contradictions
(product type, brand, capacity, GPU chip, VRAM). Manual pair decisions (work/manual_decisions.json)
are applied later in s5."""
import pandas as pd, numpy as np, re, sys, json, math
from collections import Counter
sys.path.insert(0,'work')
from normlib import *
a = pd.read_csv('work/state/s1_translated.csv', dtype=str, keep_default_na=False)
n = pd.read_csv('work/state/s2_features.csv', dtype=str, keep_default_na=False)
d = a.merge(n, on=['id','source']); idx={x:i for i,x in enumerate(d.id)}
c = pd.read_csv('work/state/s3_candidates_all.csv', dtype=str, keep_default_na=False)
GENERIC = set('''the and with for of in to a de la le du des und mit fur für pour con per en
graphics graphic card video vga grafikkort videokaart placa tarjeta scheda carte grafica grafikkarte gpu
hard drive drives disk disco duro dysk festplatte hdd ssd solid state internal external interne intern interno
portable storage flash usb stick pen pendrive memory thumb gb tb go to mb mbs ms mhz rpm bit bits cache buffer
sata sas nvme pcie pci express pcie3 gen3 gen4 gen gen3x4 x4 x16 16x m2 m 2 3 0 5 inch in inches 25 35 2280 retail bulk oem new
nvidia geforce amd radeon gddr5 gddr6 gddr6x gddr ddr5 ddr6 gddr5x hdmi dp displayport dvi turing ampere navi pascal polaris
black edition gaming graphics'''.split())
def toks(r):
    t = (r.title+' '+r.model).lower().replace('®',' ').replace('™',' ')
    t = re.sub(r'(\d)\s*(gb|tb|g|t|go|to)\b', r'\1gb', t)
    ws = re.findall(r'[a-z0-9]+', t)
    brands = set(ALIAS)
    return set(w for w in ws if w not in GENERIC and w not in brands and not re.fullmatch(r'\d+(gb|mhz|mb|mm|w)?', w))
T = [toks(r) for r in d.itertuples()]
df = Counter(w for s in T for w in s); N=len(d)
idf = {w: math.log(N/(1+df[w])) for w in df}
C = [set(filter(None,x.split('|'))) for x in d.codes]
cdf = Counter(w for s in C for w in s)
def f(x):
    try: return float(x)
    except: return None
cap = [f(x) for x in d.cap_gb]; vram=[f(x) for x in d.vram]
GPU_VARIANT = set('oc super ti xt evo strix rog tuf dual turbo trio ventus xs gp amp mini twin x2 windforce aorus advanced xc ultra ftw3 ftw sc ko phoenix pulse nitro dragon devil mech thicc eagle extreme xtreme master low profile lp itx armor aero jetstream gamerock stormx vision blower founders sff single fan 2x 3x x3 holo plus lite arctic storm hof'.split())
rows=[]
for r in c.itertuples():
    i,j = idx[r.id1], idx[r.id2]
    A,B = d.iloc[i], d.iloc[j]
    reasons=[]
    if A.ptype and B.ptype and A.ptype!=B.ptype: reasons.append('ptype')
    if A.brand_k and B.brand_k and A.brand_k!=B.brand_k: reasons.append('brand')
    if cap[i] and cap[j] and abs(cap[i]-cap[j])/max(cap[i],cap[j])>0.03: reasons.append('capacity')
    if A.chip and B.chip and A.chip!=B.chip: reasons.append('chip')
    if vram[i] and vram[j] and vram[i]!=vram[j]: reasons.append('vram')
    shared = C[i]&C[j]
    code_ev = max([1.0 if cdf[x]<=8 else 0.5 for x in shared], default=0)
    ti,tj = T[i],T[j]
    inter = sum(idf[w] for w in ti&tj); uni = sum(idf[w] for w in ti|tj)
    jac = inter/uni if uni else 0
    vi, vj = ti&GPU_VARIANT, tj&GPU_VARIANT
    vconf = len(vi^vj) if A.ptype=='GPU' else 0
    rows.append(dict(id1=r.id1,id2=r.id2,cross=r.cross,methods=r.methods,conflicts='|'.join(reasons),code=code_ev,jac=round(jac,3),vdiff=vconf,
                     shared_codes='|'.join(sorted(shared))))
p = pd.DataFrame(rows)
# decision rule (documented in report):
#  - any hard conflict (except brand when a rare code is shared) -> reject
#  - shared rare code -> accept
#  - otherwise accept if jac >= 0.55 (and for GPUs no variant-token difference), 
JAC_T = 0.55
def decide(r):
    conf = [x for x in r.conflicts.split('|') if x]
    if r.code>=1 and conf in ([],['brand']): return 1
    if conf: return 0
    if r.jac>=JAC_T and r.vdiff==0: return 1
    return 0
p['match'] = p.apply(decide, axis=1)
p.to_csv('work/state/s4_scored_pairs.csv', index=False)
print(p.match.value_counts(), p[p.match==1].cross.value_counts())
print(p.conflicts.value_counts().head(10))
