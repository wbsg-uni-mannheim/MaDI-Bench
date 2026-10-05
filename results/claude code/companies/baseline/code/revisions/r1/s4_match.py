"""Stage 4b: direct rule-based scoring of candidate pairs (no training, no labels).
Name evidence tier + contextual evidence (country, city, founding year, revenue/assets) -> score; threshold THR."""
import pandas as pd, numpy as np, os, sys, math, json
from collections import Counter
from rapidfuzz import fuzz
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'work')
from common import basic
THR = 0.9
DESC_DF = 10   # an extra token used in >=10 names is a descriptor (e.g. foods, bank, software), not a distinct name
df = pd.read_pickle('work/state/norm.pkl').set_index('id')
df['mkey'] = df.nkey.where(df.nkey.str.len() > 0, df.name.str.lower())
F = pd.read_pickle('work/state/features.pkl')
dfreq = Counter(t for k in df.mkey for t in set(k.split()))
N = len(df)
def idf(t): return math.log((N + 1) / (1 + dfreq.get(t, 0)))
GENERIC = set('group holdings holding industries industry international intl enterprises enterprise companies company co corp corporation inc incorporated limited ltd plc global worldwide cos grupo groupe gruppo de del da do di the and of s a sa ag nv se llc'.split())
cldr = pd.read_csv('task/input/schemamatching/CLDR_Country_Taxonomy.csv')
GEO = set()
for c in ['Country Name', 'Country Short Name', 'Country Variant Name']:
    for v in cldr[c].dropna(): GEO |= set(basic(v).split())
GEO |= set(('europe european asia asian america american americas africa african pacific usa us uk emea latin '
            'chinese japanese indian german british french italian spanish russian korean canadian australian brazilian mexican '
            'dutch swiss swedish norwegian danish finnish polish turkish arab arabian gulf middle east nordic scandinavia '
            'england scotland wales ireland irish hong kong taiwan singapore shanghai beijing london paris tokyo').split())
for v in df.city.dropna(): GEO |= set(basic(v).split()) if len(basic(v).split()) == 1 else set()
GEO -= GENERIC | set('new united national royal general first central saint san st de la le los las mobile energy power bank life'.split())

# tokens naming a country itself (e.g. 'australia', 'hellas' for Greece) - an extra token equal to the shared country is not a regional arm
OWNGEO = {}
for _, rr in cldr.iterrows():
    for c in ['Country Name', 'Country Short Name', 'Country Variant Name']:
        if pd.notna(rr[c]): OWNGEO.setdefault(rr['Country Name'], set()).update(basic(rr[c]).split())
OWNGEO.setdefault('Greece', set()).add('hellas'); OWNGEO.setdefault('Portugal', set()).add('portugal')
COMPAT = {frozenset(['China', 'Hong Kong SAR China'])}   # HK-listed Chinese companies: domicile vs HQ, not a contradiction

def strict_f(a, b):
    A, B = a.split(), b.split()
    if not A or not B: return 0.0, A, B
    if a.replace(' ', '') == b.replace(' ', ''): return 1.0, [], []
    SA, SB = set(A), set(B)
    ua = [t for t in A if t not in SB]; ub = [t for t in B if t not in SA]
    wA = sum(idf(t) for t in A); wB = sum(idf(t) for t in B)
    p = 1 - sum(idf(t) for t in ua) / wA; r = 1 - sum(idf(t) for t in ub) / wB
    return (0 if p + r == 0 else 2 * p * r / (p + r)), ua, ub
def soft_eq(ua, ub):
    """all unmatched tokens pair up as near-identical spellings (typos, transliteration)"""
    if len(ua) != len(ub) or not ua: return False
    from rapidfuzz.distance import Levenshtein
    def ok(t, u):
        return len(t) >= 4 and t[:3] == u[:3] and Levenshtein.distance(t, u) <= (1 if min(len(t), len(u)) <= 6 else 2)
    return all(any(ok(t, u) for u in ub) for t in ua)
def keys(x):
    return [k for k in [x.mkey, x.alt_key] if isinstance(k, str) and k]

KEYCOUNT = df.mkey.value_counts()
out = []
for r in F.itertuples():
    x, y = df.loc[r.id1], df.loc[r.id2]
    best = None
    for a in keys(x):
        for b in keys(y):
            f, ua, ub = strict_f(a, b)
            ua_ = [t for t in ua if t not in GENERIC]; ub_ = [t for t in ub if t not in GENERIC]
            if f == 1.0 or (not ua_ and not ub_): tier, nsc = 'exact_or_generic', (1.0 if f == 1.0 else 0.95)
            elif not ua_ or not ub_:
                extra = ua_ or ub_
                shorter = b if not ub_ else a
                base = [t for t in shorter.split() if t not in GENERIC and t not in GEO]
                if not base: tier, nsc = 'extra_token_weakbase', 0.4
                elif all(t in GEO for t in extra) and not (x.country is not None and x.country == y.country and all(t in OWNGEO.get(x.country, set()) for t in extra)):
                    tier, nsc = 'extra_geo', 0.4
                elif all(t in GEO for t in extra): tier, nsc = 'extra_geo_own_country', 0.7
                elif all(dfreq.get(t, 0) >= DESC_DF for t in extra): tier, nsc = 'extra_descriptor', 0.7
                else: tier, nsc = 'extra_token', 0.6
            elif soft_eq(ua_, ub_): tier, nsc = 'typo', 0.75
            else: tier, nsc = 'partial', 0.8 * f
            cand = (nsc, tier, ' '.join(ua), ' '.join(ub), f)
            if best is None or cand[0] > best[0]: best = cand
    nsc, tier, ua, ub, f = best
    if r.acronym or r.paren:
        if nsc < 0.7: nsc, tier = 0.7, 'acronym_or_alias'
    ev = 0.0; notes = []
    uniq = KEYCOUNT[x.mkey] <= 3 and len(x.mkey) >= 4
    if r.country_eq == 1: ev += 0.15; notes.append('country+')
    elif r.country_eq == 0 and frozenset([x.country, y.country]) in COMPAT: notes.append('country~')
    elif r.country_eq == 0: ev -= (0.1 if (tier == 'exact_or_generic' and uniq) else 0.25); notes.append('country-')
    if r.city_eq == 1: ev += 0.1; notes.append('city+')
    elif r.city_eq == 0: ev -= 0.05; notes.append('city-')
    if r.year_diff == 0: ev += 0.15; notes.append('year=')
    elif r.year_diff <= 2: ev += 0.05; notes.append('year~')
    elif r.year_diff > 10: ev -= 0.1; notes.append('year!')
    lrs = [v for v in [r.rev_lr, r.assets_lr] if not pd.isna(v)]
    if lrs and min(lrs) < 0.05: ev += 0.2; notes.append('money=')
    elif lrs and min(lrs) < 0.15: ev += 0.1; notes.append('money~')
    elif lrs and min(lrs) > 1: ev -= 0.1; notes.append('money!')
    out.append(dict(id1=r.id1, id2=r.id2, src1=r.src1, src2=r.src2, n1=r.n1, n2=r.n2, tier=tier, name_score=nsc, strict_f=round(f, 3),
                    un1=ua, un2=ub, evidence=round(ev, 3), notes=' '.join(notes), score=round(nsc + ev, 3)))
M = pd.DataFrame(out)
M['match'] = M.score >= THR
M.to_csv('work/state/scored_pairs.csv', index=False)
print(M.groupby(['tier'])['match'].agg(['size', 'sum']))
print(M[M.match].groupby(M.src1 + '-' + M.src2).size())
