"""Stage 3: blocking. Union of complementary candidate generators over normalized records.
Writes work/state/candidates.csv (id1,id2,sources) and blocking diagnostics."""
import pandas as pd, numpy as np, os, json, re, itertools
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
n = pd.read_pickle(f'{W}/state/normalized.pkl')
ids = n.id.tolist(); src = n.source.tolist()
pairs = defaultdict(set)
def add(i, j, why):
    if i == j: return
    a, b = (i, j) if ids[i] < ids[j] else (j, i)
    pairs[(a, b)].add(why)

# (a) shared identifier-like code (title/model/model_number); skip very common codes
code_idx = defaultdict(list)
for i, cs in enumerate(n.n_codes):
    for c in cs: code_idx[c].append(i)
for c, L in code_idx.items():
    if len(L) > 40: continue
    for i, j in itertools.combinations(L, 2): add(i, j, 'code')
# (b) same normalized model_number field
mn_idx = defaultdict(list)
for i, m in enumerate(n.n_mn):
    if m and len(m) >= 4: mn_idx[m].append(i)
for m, L in mn_idx.items():
    if len(L) > 40: continue
    for i, j in itertools.combinations(L, 2): add(i, j, 'mn')
# (c) TF-IDF char n-gram nearest neighbours on title+model (k=15)
text = (n.title + ' ' + n.model).str.lower().str.replace(r'[^a-z0-9. ]', ' ', regex=True)
X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform(text)
K = 15
nn = NearestNeighbors(n_neighbors=K + 1, metric='cosine').fit(X)
dist, ind = nn.kneighbors(X)
for i in range(len(ids)):
    for d, j in zip(dist[i], ind[i]):
        if j != i and d < 0.6: add(i, j, 'tfidf')
# (d) spec key: (ptype, brand, capacity) for storage, (brand, chip, vram) for GPU; capped block size
key_idx = defaultdict(list)
for i, r in n.iterrows():
    if r.n_product_type in ('SSD', 'HDD', 'USB_STICK') and r.n_brand and pd.notna(r.n_cap_gb):
        key_idx[('S', r.n_brand, round(r.n_cap_gb))].append(i)
    if r.n_product_type == 'GPU' and r.n_brand and r.n_chip:
        key_idx[('G', r.n_brand, r.n_chip)].append(i)
big = {}
for k, L in key_idx.items():
    if len(L) > 60: big[str(k)] = len(L); continue
    for i, j in itertools.combinations(L, 2): add(i, j, 'speckey')

rows = [(ids[a], ids[b], '|'.join(sorted(w))) for (a, b), w in pairs.items()]
c = pd.DataFrame(rows, columns=['id1', 'id2', 'why'])
c.to_csv(f'{W}/state/candidates.csv', index=False)
N = len(ids); total = N * (N - 1) // 2
per = Counter(); 
for a, b in pairs: per[a] += 1; per[b] += 1
diag = dict(n_candidates=len(c), total_pairs=total, reduction_ratio=round(1 - len(c) / total, 5),
            by_method={m: int(c.why.str.contains(m).sum()) for m in ['code', 'mn', 'tfidf', 'speckey']},
            unique_to_method={m: int((c.why == m).sum()) for m in ['code', 'mn', 'tfidf', 'speckey']},
            zero_candidate_records=int(sum(1 for i in range(N) if per[i] == 0)),
            max_candidates_per_record=int(max(per.values())), median_candidates=float(np.median([per[i] for i in range(N)])),
            skipped_big_spec_blocks=big,
            cross_source_share=round(float(np.mean([src[a] != src[b] for a, b in pairs])), 4))
print(json.dumps(diag, indent=1))
json.dump(diag, open(f'{W}/state/blocking_diag.json', 'w'), indent=1)
