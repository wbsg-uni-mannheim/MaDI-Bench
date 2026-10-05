"""Stage 3: blocking.
Title level: union of (a) char-3gram TF-IDF top-k neighbours, (b) sorted-token key, (c) generic-token-stripped key.
Record level: cross-source record pairs whose titles are candidate-linked (or equal) and whose platforms are
compatible (same platform key, or at least one side unknown).
Outputs: state/title_pairs.csv, state/candidates.csv (id1,id2), submission/blocking/candidates.csv
"""
import os, json, time, collections, itertools
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from common import initials_key, acronym_key

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["dbpedia", "metacritic", "sales"]
TOPK, MINCOS = 15, 0.35
GENERIC = {"the", "soccer", "video", "game", "videogame", "edition", "a"}

recs = pd.concat([pd.read_csv(f"{W}/state/norm_{s}.csv", dtype=str, keep_default_na=False) for s in SRCS],
                 ignore_index=True)
titles = sorted(set(recs["ntitle"]) - {""})
tidx = {t: i for i, t in enumerate(titles)}
print("unique titles", len(titles))

pairs = set()
# (a) tf-idf char n-grams
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
X = vec.fit_transform(titles).tocsr()
t0 = time.time()
for st in range(0, X.shape[0], 2000):
    S = (X[st:st + 2000] @ X.T).toarray()
    for r in range(S.shape[0]):
        i = st + r
        row = S[r]
        row[i] = 0
        k = min(TOPK, len(row) - 1)
        nb = np.argpartition(-row, k)[:k]
        for j in nb:
            if row[j] >= MINCOS:
                pairs.add((min(i, j), max(i, j)))
print("tfidf pairs", len(pairs), round(time.time() - t0, 1), "s")
# (b) sorted-token key, (c) generic-stripped sorted key
for keyf in (lambda t: " ".join(sorted(t.split())),
             lambda t: " ".join(sorted(w for w in t.split() if w not in GENERIC))):
    groups = collections.defaultdict(list)
    for t in titles:
        k = keyf(t)
        if k:
            groups[k].append(tidx[t])
    n0 = len(pairs)
    for g in groups.values():
        for a, b in itertools.combinations(sorted(g), 2):
            pairs.add((a, b))
    print("key pairs added", len(pairs) - n0)
# (e) abbreviation key: 'GTA IV' ~ 'grand theft auto iv' (initials of the full title, letters of the acronym)
full = collections.defaultdict(set)
for t in titles:
    if len(t.split()) >= 2:
        full[initials_key(t)].add(tidx[t])
acr_pairs = set()
for raw, nt in zip(recs["name"], recs["ntitle"]):
    if not nt:
        continue
    k = acronym_key(raw)
    if k and len(k) >= 3 and k in full:
        for j in full[k]:
            if j != tidx[nt]:
                acr_pairs.add((min(tidx[nt], j), max(tidx[nt], j)))
print("acronym title pairs", len(acr_pairs), "new", len(acr_pairs - pairs))
pairs |= acr_pairs
pd.DataFrame([(titles[i], titles[j]) for i, j in sorted(acr_pairs)], columns=["t1", "t2"]).to_csv(
    f"{W}/state/acronym_pairs.csv", index=False)
tp = pd.DataFrame(sorted(pairs), columns=["i", "j"])
tp["t1"] = [titles[i] for i in tp.i]; tp["t2"] = [titles[j] for j in tp.j]
tp[["t1", "t2"]].to_csv(f"{W}/state/title_pairs.csv", index=False)

# record-level candidates
nbr = collections.defaultdict(set)
for a, b in pairs:
    nbr[a].add(b); nbr[b].add(a)
by_title = collections.defaultdict(list)
for r in recs.itertuples(index=False):
    if r.ntitle:
        by_title[tidx[r.ntitle]].append((r.id, r.source, r.pkey))


def compat(p, q):
    return p == "" or q == "" or p == q


cand = set()
for ti, rl in by_title.items():
    others = [ti] + [tj for tj in nbr[ti] if tj > ti]
    for tj in others:
        rl2 = by_title.get(tj, [])
        for (i1, s1, p1) in rl:
            for (i2, s2, p2) in rl2:
                if s1 != s2 and compat(p1, p2):
                    cand.add((i1, i2) if i1 < i2 else (i2, i1))
# (d) records without a title (metacritic/sales): attribute-key block on (platform, critic score)
ms = recs[recs.source.isin(["metacritic", "sales"]) & (recs.pkey != "") & (recs.critic != "")]
n0 = len(cand)
for (pk, cs), g in ms.groupby(["pkey", "critic"]):
    if (g.ntitle == "").sum() == 0 or g.source.nunique() < 2:
        continue
    for a in g[g.ntitle == ""].itertuples(index=False):
        for b in g[g.source != a.source].itertuples(index=False):
            cand.add((a.id, b.id) if a.id < b.id else (b.id, a.id))
print("attribute-key pairs for untitled records", len(cand) - n0)
cdf = pd.DataFrame(sorted(cand), columns=["id1", "id2"])
cdf.to_csv(f"{W}/state/candidates.csv", index=False)
os.makedirs(f"{ROOT}/submission/blocking", exist_ok=True)
cdf.to_csv(f"{ROOT}/submission/blocking/candidates.csv", index=False)

# diagnostics
src_of = dict(zip(recs.id, recs.source))
sp = collections.Counter(tuple(sorted((src_of[a], src_of[b]))) for a, b in cand)
covered = collections.Counter()
seen = set(cdf.id1) | set(cdf.id2)
for r in recs.itertuples(index=False):
    covered[(r.source, r.id in seen)] += 1
sizes = {s: int((recs.source == s).sum()) for s in SRCS}
full = sizes["dbpedia"] * sizes["metacritic"] + sizes["dbpedia"] * sizes["sales"] + sizes["metacritic"] * sizes["sales"]
diag = dict(stage="blocking", title_pairs=len(pairs), candidates=len(cand),
            by_source_pair={f"{a}-{b}": c for (a, b), c in sp.items()},
            records_with_candidate={s: covered[(s, True)] for s in SRCS},
            records_without_candidate={s: covered[(s, False)] for s in SRCS},
            reduction_ratio=round(1 - len(cand) / full, 6), full_cross=full,
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs="state/norm_*.csv")
print(json.dumps(diag, indent=1))
with open(f"{W}/diagnostics.jsonl", "a") as f:
    f.write(json.dumps(diag) + "\n")
