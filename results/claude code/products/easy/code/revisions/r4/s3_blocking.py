"""Stage 3: blocking = union of (a) equal normalized model_number, (b) shared rare identifier code,
(c) TF-IDF title nearest neighbours (top-K per record per other source)."""
import os, json, collections, itertools
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
K, MAX_CODE_DF = 5, 16
d = pd.read_pickle('work/state/norm.pkl')
ids, srcs = d.id.values, d.source.values
pairs = collections.defaultdict(set)
def add(i, j, how):
    if i == j: return
    a, b = (i, j) if ids[i] < ids[j] else (j, i)
    pairs[(a, b)].add(how)
# (a) model number key
for k, g in d.groupby('mn_key').groups.items():
    g = list(g)
    for i, j in itertools.combinations(g, 2): add(i, j, 'mn')
# (b) rare code tokens
inv = collections.defaultdict(list)
for i, cs in enumerate(d.codes):
    for c in cs: inv[c].append(i)
big = 0
for c, g in inv.items():
    if len(g) > MAX_CODE_DF: big += 1; continue
    for i, j in itertools.combinations(g, 2): add(i, j, 'code')
# (c) tfidf neighbours across sources (word + char)
tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1, sublinear_tf=True)
X = tf.fit_transform(d.title_n + ' ' + d.brand.fillna('').str.lower())
for s in sorted(set(srcs)):
    tgt = np.where(srcs == s)[0]
    nn = NearestNeighbors(n_neighbors=K, metric='cosine').fit(X[tgt])
    dist, idx = nn.kneighbors(X)
    for i in range(len(d)):
        for dd, jj in zip(dist[i], idx[i]):
            j = tgt[jj]
            if 1 - dd >= 0.35: add(i, j, 'tfidf')
rows = [(ids[a], ids[b], a, b, '|'.join(sorted(h))) for (a, b), h in pairs.items()]
cand = pd.DataFrame(rows, columns=['id1', 'id2', 'i1', 'i2', 'how'])
cand.to_pickle('work/state/candidates.pkl')
n = len(d); total = n * (n - 1) // 2
cnt = collections.Counter(); [cnt.update([a, b]) for a, b in pairs]
stats = {'candidates': len(cand), 'all_pairs': total, 'reduction_ratio': 1 - len(cand) / total,
         'by_method': cand.how.value_counts().to_dict(), 'skipped_frequent_codes': big,
         'records_without_candidates': int(n - len(cnt)), 'max_cands_per_record': max(cnt.values()),
         'cross_source_share': float((srcs[cand.i1] != srcs[cand.i2]).mean())}
print(json.dumps(stats)); json.dump(stats, open('work/state/blocking_stats.json', 'w'))
