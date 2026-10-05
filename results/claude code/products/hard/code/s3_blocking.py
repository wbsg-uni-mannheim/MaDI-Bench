"""Stage 3: blocking. Union of (A) spec-key blocks, (B) shared rare model codes, (C) TF-IDF title kNN.
All pairs are between records (any sources); cross-source pairs are exported for submission,
within-source pairs are kept internally because the same product can occur twice in one source."""
import pandas as pd, numpy as np, re, sys, json, itertools
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
a = pd.read_csv('work/state/s1_translated.csv', dtype=str, keep_default_na=False)
n = pd.read_csv('work/state/s2_features.csv', dtype=str, keep_default_na=False)
d = a[['id','source','title']].merge(n, on=['id','source'])
ids = d.id.tolist(); idx = {x:i for i,x in enumerate(ids)}
pairs = defaultdict(set)
def add(i,j,m):
    if i==j: return
    if i>j: i,j=j,i
    pairs[(i,j)].add(m)
# A: spec key; unknown brand joins every brand within same ptype+spec
spec = d.apply(lambda r: r.chip if r.ptype=='GPU' else (r.cap_gb if r.cap_gb else ''), axis=1)
d['spec']=spec
grp = defaultdict(list)
for i,r in d.iterrows():
    if r.ptype and r.spec: grp[(r.ptype, r.spec)].append(i)
for k, mem in grp.items():
    for i,j in itertools.combinations(mem,2):
        bi, bj = d.brand_k[i], d.brand_k[j]
        if bi==bj or not bi or not bj: add(i,j,'spec')
# B: shared rare codes
cidx = defaultdict(list)
for i,c in enumerate(d.codes):
    for t in filter(None, c.split('|')): cidx[t].append(i)
for t, mem in cidx.items():
    if 2 <= len(mem) <= 15:
        for i,j in itertools.combinations(mem,2): add(i,j,'code')
# C: TF-IDF kNN on title (char 3-5 grams), k=8
tv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1, sublinear_tf=True, lowercase=True)
X = tv.fit_transform(d.title.str.replace(r'[^\w\s.]',' ',regex=True))
K=8
nn = NearestNeighbors(n_neighbors=K+1, metric='cosine').fit(X)
dist, nb = nn.kneighbors(X)
for i in range(len(d)):
    for dd,j in zip(dist[i][1:], nb[i][1:]):
        if 1-dd >= 0.35: add(i,j,'knn')
rows=[dict(id1=ids[i], id2=ids[j], methods='+'.join(sorted(m))) for (i,j),m in pairs.items()]
c = pd.DataFrame(rows)
c['cross'] = [d.source[idx[x]]!=d.source[idx[y]] for x,y in zip(c.id1,c.id2)]
c.to_csv('work/state/s3_candidates_all.csv', index=False)
N=len(d); srcs=d.source.value_counts()
full_cross = sum(srcs[a_]*srcs[b_] for a_,b_ in itertools.combinations(srcs.index,2))
cc = c[c.cross]
stats = dict(total_pairs=len(c), cross_pairs=len(cc), full_cross=int(full_cross), reduction_ratio=1-len(cc)/full_cross,
             by_method={m:int(cc.methods.str.contains(m).sum()) for m in ['spec','code','knn']},
             unique_to={m:int((cc.methods==m).sum()) for m in ['spec','code','knn']},
             records_without_cross_candidate=int(N-len(set(cc.id1)|set(cc.id2))),
             largest_spec_block=max(len(v) for v in grp.values()))
print(json.dumps(stats, indent=1))
json.dump(stats, open('work/state/s3_stats.json','w'), indent=1)
