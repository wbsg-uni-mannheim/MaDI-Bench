"""Stage 3: blocking. Cross-source candidate pairs:
 A) same canonical platform key AND fuzzy name similarity >= 70 (token_set_ratio or token_sort_ratio)
 B) at least one side has no/unresolved platform AND name similarity >= 85
Within-dbpedia pairs (for duplicate grouping) are produced by the same rule A and saved separately."""
import pandas as pd, numpy as np, os, time
from rapidfuzz import process, fuzz
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
U = pd.read_pickle(f'{BASE}/work/state/s2_norm.pkl')
TH_A, TH_B = 70, 85
def sim_pairs(A, B, th, same=False):
    """rapidfuzz cdist over nkey; returns list of (i, j, set_ratio, sort_ratio) index pairs into A, B."""
    if len(A) == 0 or len(B) == 0: return []
    a, b = A.nkey.tolist(), B.nkey.tolist()
    s1 = process.cdist(a, b, scorer=fuzz.token_set_ratio, score_cutoff=th, dtype=np.uint8)
    s2 = process.cdist(a, b, scorer=fuzz.token_sort_ratio, score_cutoff=th, dtype=np.uint8)
    ii, jj = np.nonzero(np.maximum(s1, s2))
    if same: keep = ii < jj; ii, jj = ii[keep], jj[keep]
    return [(A.index[i], B.index[j], int(s1[i, j]), int(s2[i, j])) for i, j in zip(ii, jj)]
t = time.time(); out = []; dd = []
srcs = ['metacritic', 'sales', 'dbpedia']
for pk, g in U[U.pkb.notna()].groupby('pkb'):
    parts = {s: g[g.source == s] for s in srcs}
    for x in range(3):
        for y in range(x + 1, 3):
            out += [(i, j, a, b, 'A') for i, j, a, b in sim_pairs(parts[srcs[x]], parts[srcs[y]], TH_A)]
    dd += sim_pairs(parts['dbpedia'], parts['dbpedia'], 85, same=True)
print('A done', len(out), time.time() - t)
miss = U[U.pkb.isna()]
for s in srcs:
    ms = miss[miss.source == s]
    for s2 in srcs:
        if s2 == s: continue
        other = U[U.source == s2]
        # compare only to other-source records where this pair is not already covered by rule A
        out += [(i, j, a, b, 'B') for i, j, a, b in sim_pairs(ms, other, TH_B)]
print('B done', len(out), time.time() - t)
C = pd.DataFrame(out, columns=['i', 'j', 'set_r', 'sort_r', 'rule'])
C['a'] = np.minimum(C.i, C.j); C['b'] = np.maximum(C.i, C.j)
C = C.sort_values('rule').drop_duplicates(['a', 'b'])[['a', 'b', 'set_r', 'sort_r', 'rule']].reset_index(drop=True)
C.to_pickle(f'{BASE}/work/state/s3_candidates.pkl')
pd.DataFrame(dd, columns=['a', 'b', 'set_r', 'sort_r']).to_pickle(f'{BASE}/work/state/s3_dbp_internal.pkl')
os.makedirs(f'{BASE}/submission/blocking', exist_ok=True)
pd.DataFrame({'id1': U.rid.values[C.a], 'id2': U.rid.values[C.b]}).to_csv(f'{BASE}/submission/blocking/candidates.csv', index=False)
print(len(C), C.rule.value_counts().to_dict(), 'dbp-internal', len(dd), time.time() - t)
