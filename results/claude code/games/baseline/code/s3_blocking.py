"""Stage 3: build matching units (dbpedia rows grouped by exact title incl. disambiguation + platform key)
and generate cross-source candidate unit pairs within the same platform key."""
import pandas as pd, numpy as np, sys, json, re
sys.path.insert(0,'work')
from norm import *
from rapidfuzz import process, fuzz
N = pd.read_pickle('work/state/normalized.pkl')
# --- units
db = N[N.source=='dbpedia'].copy()
# unit = identical (title incl. disambiguation, platform key, release year): rows of the dbpedia cross-product that
# assert the same game, platform AND date. Rows of one title/platform with different dates are kept apart (their
# release information contradicts each other); matching links the date-consistent unit.
db['gkey'] = db.name_raw.map(lambda s: ' '.join(title_tokens(s))) + '||' + db.pkey.fillna('NONE') + '||' + db.year.map(lambda y: 'NA' if pd.isna(y) else str(int(y)))
gid = {k:f'G{i}' for i,k in enumerate(sorted(db.gkey.unique()))}
db['unit'] = db.gkey.map(gid)
N['unit'] = N.record_id
N.loc[db.index,'unit'] = db.unit
N.to_pickle('work/state/normalized_units.pkl')
agg = N.groupby('unit').agg(source=('source','first'), tkey=('tkey','first'), pkey=('pkey','first'),
        name=('name','first'), years=('year', lambda s: sorted(set(int(x) for x in s.dropna()))),
        dyear=('disamb_year','first'), critic=('criticScore','first'), user=('userScore','first'), n=('record_id','size')).reset_index()
agg.to_pickle('work/state/units.pkl')
print(agg.source.value_counts().to_dict(), 'dbpedia rows/unit', agg[agg.source=='dbpedia'].n.describe().round(2).to_dict())
# --- candidates
TOPK, MINSIM = 8, 60
cands = set(); method = {}
def add(a,b,m):
    k = tuple(sorted((a,b)))
    cands.add(k); method.setdefault(k,set()).add(m)
U = {s:g for s,g in agg[agg.pkey.notna()].groupby('source')}
for s1, s2 in [('metacritic','sales'),('metacritic','dbpedia'),('sales','dbpedia')]:
    A, B = U[s1], U[s2]
    # exact title key
    for a,b in A.merge(B, on=['tkey','pkey'])[['unit_x','unit_y']].itertuples(index=False): add(a,b,'exact')
    # fuzzy within platform: top-k by token_set_ratio and by WRatio
    for p, ga in A.groupby('pkey'):
        gb = B[B.pkey==p]
        if gb.empty: continue
        bl = gb.tkey.tolist(); bu = gb.unit.tolist()
        for scorer, tag in [(fuzz.token_set_ratio,'tset'), (fuzz.ratio,'ratio')]:
            M = process.cdist(ga.tkey.tolist(), bl, scorer=scorer, workers=-1)
            k = min(TOPK, M.shape[1])
            idx = np.argpartition(-M, k-1, axis=1)[:, :k]
            for i, a in enumerate(ga.unit.tolist()):
                for j in idx[i]:
                    if M[i,j] >= MINSIM: add(a, bu[j], tag)
            # reverse direction top-k
            idx2 = np.argpartition(-M.T, min(TOPK, M.shape[0])-1, axis=1)[:, :min(TOPK, M.shape[0])]
            au = ga.unit.tolist()
            for j, b in enumerate(bu):
                for i in idx2[j]:
                    if M[i,j] >= MINSIM: add(au[i], b, tag)
C = pd.DataFrame([(a,b,'|'.join(sorted(method[(a,b)]))) for a,b in sorted(cands)], columns=['u1','u2','methods'])
C.to_pickle('work/state/candidates_units.pkl')
src = agg.set_index('unit').source
C['pair'] = C.u1.map(src)+'-'+C.u2.map(src)
print(C.pair.value_counts().to_dict()); print(C.methods.value_counts().head(8).to_dict())
