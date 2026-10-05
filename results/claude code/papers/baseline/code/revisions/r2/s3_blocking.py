"""Stage 3: blocking. Union of (A) exact title key, (B) TF-IDF title cosine top-k per source pair,
(C) author-surname keys (first+last surname+year, full surname set)."""
import os, json, itertools, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
N = pd.read_pickle('work/state/s2_normalized.pkl')
SRC = ['crossref','dblp','open_alex']
PAIRS = list(itertools.combinations(SRC, 2))
MAXBLOCK = 30  # skip key blocks with more than this many records per source
K, MINSIM = 5, 0.3
idx = {s: N.index[N.source == s].to_numpy() for s in SRC}
cands = defaultdict(set)  # (i,j) -> set of methods, i<j global row idx

def key_block(keys, name):
    groups = defaultdict(lambda: defaultdict(list))
    for i, k in keys.items():
        if k: groups[k][N.at[i, 'source']].append(i)
    for k, g in groups.items():
        if any(len(v) > MAXBLOCK for v in g.values()): continue
        for a, b in PAIRS:
            for i in g.get(a, []):
                for j in g.get(b, []):
                    cands[(min(i,j), max(i,j))].add(name)

# A exact title key (at least 12 chars to avoid 'poster','sok')
key_block(N['tkey'].where(N['tkey'].str.len() >= 12), 'title_exact')
# C author keys
def fl(r):
    s = r.surnames
    if len(s) < 2: return None
    return f'{s[0]}|{s[-1]}|{r.publication_year}'
key_block(pd.Series([fl(r) for r in N.itertuples()], index=N.index), 'auth_firstlast_year')
key_block(N['surnames'].map(lambda s: '|'.join(sorted(s)) if len(s) >= 3 else None), 'auth_set')
# single-author + title prefix
key_block(pd.Series([f'{r.surnames[0]}|{" ".join(r.ttok[:2])}' if r.surnames and len(r.ttok)>=2 else None for r in N.itertuples()], index=N.index), 'auth1_title2')
# B tfidf
vec = TfidfVectorizer(analyzer='word', token_pattern=r'[a-z0-9]+', sublinear_tf=True, min_df=1)
texts = N['ttok'].map(' '.join)
X = vec.fit_transform(texts)
for a, b in PAIRS:
    Xa, Xb = X[idx[a]], X[idx[b]].T.tocsc()
    for st in range(0, Xa.shape[0], 4000):
        S = (Xa[st:st+4000] @ Xb).toarray()
        top = np.argpartition(-S, K, axis=1)[:, :K]
        for r in range(S.shape[0]):
            i = idx[a][st + r]
            for c in top[r]:
                if S[r, c] >= MINSIM:
                    j = idx[b][c]; cands[(min(i,j), max(i,j))].add('tfidf_topk')
    # reverse direction top-k too
    Xb2, Xa2 = X[idx[b]], X[idx[a]].T.tocsc()
    for st in range(0, Xb2.shape[0], 4000):
        S = (Xb2[st:st+4000] @ Xa2).toarray()
        top = np.argpartition(-S, K, axis=1)[:, :K]
        for r in range(S.shape[0]):
            i = idx[b][st + r]
            for c in top[r]:
                if S[r, c] >= MINSIM:
                    j = idx[a][c]; cands[(min(i,j), max(i,j))].add('tfidf_topk')
    print('done', a, b, len(cands), flush=True)
C = pd.DataFrame([(i, j, '+'.join(sorted(m))) for (i, j), m in cands.items()], columns=['i','j','methods'])
C['id1'] = N['id'].to_numpy()[C.i]; C['id2'] = N['id'].to_numpy()[C.j]
C.to_pickle('work/state/s3_candidates.pkl')
os.makedirs('submission/blocking', exist_ok=True)
C[['id1','id2']].to_csv('submission/blocking/candidates.csv', index=False)
# diagnostics
C['sp'] = N['source'].to_numpy()[C.i] + '~' + N['source'].to_numpy()[C.j]
deg = pd.concat([C.i, C.j]).value_counts()
total = sum(len(idx[a])*len(idx[b]) for a,b in PAIRS)
diag = dict(stage='blocking', n_candidates=len(C), by_pair=C.sp.value_counts().to_dict(),
            by_method={m: int(C.methods.str.contains(m).sum()) for m in ['title_exact','tfidf_topk','auth_firstlast_year','auth_set','auth1_title2']},
            only_method=C.methods.value_counts().head(10).to_dict(),
            zero_candidate_records={s: int((~pd.Series(idx[s]).isin(deg.index)).sum()) for s in SRC},
            max_deg=int(deg.max()), reduction_ratio=1 - len(C)/total)
print(json.dumps(diag, indent=1))
json.dump(diag, open('work/state/s3_diag.json','w'))
