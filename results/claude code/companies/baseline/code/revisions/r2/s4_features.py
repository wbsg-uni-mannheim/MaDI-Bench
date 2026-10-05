"""Stage 4a: pairwise evidence for every blocking candidate."""
import pandas as pd, numpy as np, os, sys, math
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'work')
from common import basic
df = pd.read_pickle('work/state/norm.pkl').set_index('id')
df['mkey'] = df.nkey.where(df.nkey.str.len() > 0, df.name.str.lower())
cand = pd.read_csv('work/state/candidates.csv')
# token document frequency over all names (for rarity-weighted overlap)
from collections import Counter
dfreq = Counter(t for k in df.mkey for t in set(k.split()))
N = len(df)
def idf(t): return math.log(N / (1 + dfreq.get(t, 0)))
def wjacc(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B: return 0.0
    inter = sum(idf(t) for t in A & B); uni = sum(idf(t) for t in A | B)
    return inter / uni if uni else 0.0
def extra_tokens(a, b):
    A, B = a.split(), b.split()
    return [t for t in A if t not in B], [t for t in B if t not in A]
rows = []
for r in cand.itertuples():
    x, y = df.loc[r.id1], df.loc[r.id2]
    keys_x = [k for k in [x.mkey, x.alt_key] if isinstance(k, str) and k]
    keys_y = [k for k in [y.mkey, y.alt_key] if isinstance(k, str) and k]
    best = max(((fuzz.ratio(a, b) / 100, a, b) for a in keys_x for b in keys_y), key=lambda t: t[0])
    a, b = best[1], best[2]
    ex, ey = extra_tokens(a, b)
    f = dict(id1=r.id1, id2=r.id2, src1=x.source, src2=y.source, how=r.how, n1=x['name'], n2=y['name'],
             exact=float(any(k1 == k2 for k1 in keys_x for k2 in keys_y)),
             core_eq=float(x.ckey == y.ckey and len(x.ckey) > 0),
             ratio=best[0], jw=JaroWinkler.similarity(a, b), tset=fuzz.token_set_ratio(a, b) / 100,
             wj=max(wjacc(k1, k2) for k1 in keys_x for k2 in keys_y),
             extra1=' '.join(ex), extra2=' '.join(ey),
             acronym=float('acronym' in r.how), paren=float('paren' in r.how))
    f['country_eq'] = np.nan if (x.country is None or y.country is None) else float(x.country == y.country)
    f['city_eq'] = np.nan if (x.city is None or y.city is None) else float(basic(x.city) == basic(y.city) or basic(x.city) in basic(y.city) or basic(y.city) in basic(x.city))
    f['year_diff'] = np.nan if (pd.isna(x.founded) or pd.isna(y.founded)) else abs(x.founded - y.founded)
    f['ind_eq'] = np.nan if (x.industry is None or y.industry is None) else float(x.industry == y.industry)
    def lr(u, v):
        if pd.isna(u) or pd.isna(v) or u <= 0 or v <= 0: return np.nan
        return abs(math.log10(u / v))
    f['assets_lr'] = lr(x.assets, y.assets); f['rev_lr'] = lr(x.revenue, y.revenue)
    rows.append(f)
F = pd.DataFrame(rows)
F.to_pickle('work/state/features.pkl')
print(F.describe().T)
