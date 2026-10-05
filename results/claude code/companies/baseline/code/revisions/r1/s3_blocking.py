"""Stage 3: blocking. Union of (a) exact normalized-name keys, (b) TF-IDF char-ngram kNN in both directions per source pair,
(c) acronym key, (d) parenthetical-alias keys. Also within-fullcontact/within-source exact-key pairs (duplicate rows)."""
import pandas as pd, numpy as np, os, sys, itertools, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'work')
K = 10
df = pd.read_pickle('work/state/norm.pkl').reset_index(drop=True)
df['mkey'] = df.nkey.where(df.nkey.str.len() > 0, df.name.str.lower())
def acronym(k):
    t = [w for w in k.split() if w not in ('and', 'of', 'the', 'de')]
    return ''.join(w[0] for w in t) if len(t) >= 2 else None
df['acr'] = df.mkey.map(acronym)
pairs = {}
def add(i, j, how):
    if i == j: return
    a, b = (i, j) if i < j else (j, i)
    pairs.setdefault((a, b), set()).add(how)
srcs = ['dbpedia', 'forbes', 'fullcontact']
# (a) exact keys (across and within sources)
for key in ['mkey', 'ckey', 'alt_key']:
    idx = {}
    for i, v in df[key].items():
        if isinstance(v, str) and len(v) >= 2: idx.setdefault(v, []).append(i)
    # also index alt keys against mkey space
    for v, ids in idx.items():
        others = df.index[df.mkey == v].tolist() if key == 'alt_key' else []
        allids = sorted(set(ids + others))
        if len(allids) > 30: continue
        for i, j in itertools.combinations(allids, 2): add(i, j, 'exact_' + key)
# (c) acronym: short all-caps-ish names in one source vs initials of other names
short = df[df.mkey.str.fullmatch(r'[a-z]{2,6}')]
acr = {}
for i, v in df.acr.items():
    if v: acr.setdefault(v, []).append(i)
for i, v in short.mkey.items():
    for j in acr.get(v, [])[:30]:
        if df.source[i] != df.source[j]: add(i, j, 'acronym')
# (d) parenthetical aliases
for i, p in df.paren.items():
    for a in (p.split('|') if p else []):
        if len(a) >= 3:
            for j in df.index[(df.mkey == a) | (df.ckey == a)].tolist()[:30]:
                if df.source[i] != df.source[j]: add(i, j, 'paren')
# (b) tf-idf kNN
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=1)
X = vec.fit_transform(df.mkey)
for s1, s2 in itertools.combinations(srcs, 2):
    for a, b in [(s1, s2), (s2, s1)]:
        ia = df.index[df.source == a].values; ib = df.index[df.source == b].values
        nn = NearestNeighbors(n_neighbors=K, metric='cosine').fit(X[ib])
        dist, ind = nn.kneighbors(X[ia])
        for r, i in enumerate(ia):
            for c in range(K):
                if 1 - dist[r, c] >= 0.3: add(i, ib[ind[r, c]], 'knn')
# token-overlap tfidf over words using alt names too (captures reordered words)
cand = pd.DataFrame([(df.id[a], df.id[b], df.source[a], df.source[b], '|'.join(sorted(h))) for (a, b), h in pairs.items()],
                    columns=['id1', 'id2', 'src1', 'src2', 'how'])
cand.to_csv('work/state/candidates.csv', index=False)
os.makedirs('submission/blocking', exist_ok=True)
cand[['id1', 'id2']].to_csv('submission/blocking/candidates.csv', index=False)
# diagnostics
tot = {('dbpedia','forbes'): 10085*2000, ('dbpedia','fullcontact'): 10085*1931, ('forbes','fullcontact'): 2000*1931}
cand['sp'] = [tuple(sorted(x)) for x in zip(cand.src1, cand.src2)]
diag = {'n_candidates': len(cand), 'by_pair': {'/'.join(k): int(v) for k, v in cand.sp.value_counts().items()},
        'reduction_ratio': {'/'.join(k): round(1 - (cand.sp == k).sum() / v, 5) for k, v in tot.items()},
        'by_method': {m: int(cand.how.str.contains(m).sum()) for m in ['exact_mkey', 'exact_ckey', 'exact_alt_key', 'acronym', 'paren', 'knn']}}
ids = set(cand.id1) | set(cand.id2)
diag['records_without_candidates'] = {s: int((~df[df.source == s].id.isin(ids)).sum()) for s in srcs}
print(json.dumps(diag, indent=1))
json.dump(diag, open('work/state/blocking_diag.json', 'w'), indent=1)
