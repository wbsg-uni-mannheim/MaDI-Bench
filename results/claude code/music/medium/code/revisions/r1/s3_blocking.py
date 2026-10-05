"""Stage 3: blocking. Union of complementary candidate generators per source pair:
 A) top-k nearest neighbours on char-3gram TF-IDF of the normalized title
 B) top-k nearest neighbours on TF-IDF of title + artist (disambiguates generic titles)
 C) shared rare normalized track titles (>=2 shared, or 1 shared for 1-2 track releases)
Writes work/state/candidates.pkl and submission/blocking/candidates.csv."""
import pandas as pd, numpy as np, os, json, datetime
from collections import defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["discogs", "lastfm", "musicbrainz"]
K_NAME, K_NA, MIN_COS = 10, 10, 0.35

def topk_pairs(Xa, Xb, k, min_cos):
    out = []
    step = 2000
    for i0 in range(0, Xa.shape[0], step):
        S = (Xa[i0:i0 + step] @ Xb.T).toarray()
        idx = np.argpartition(-S, min(k, S.shape[1] - 1), axis=1)[:, :k]
        for r in range(S.shape[0]):
            for j in idx[r]:
                if S[r, j] >= min_cos:
                    out.append((i0 + r, int(j), float(S[r, j])))
    return out

def main():
    N = {s: pd.read_pickle(f"{W}/state/norm_{s}.pkl").reset_index(drop=True) for s in SRCS}
    allnames = pd.concat([N[s].name_key for s in SRCS])
    na = {s: (N[s].name_key + " | " + N[s].artist_key.fillna("")).str.strip() for s in SRCS}
    v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True).fit(allnames)
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True).fit(pd.concat(na.values()))
    X1 = {s: v1.transform(N[s].name_key) for s in SRCS}
    X2 = {s: v2.transform(na[s]) for s in SRCS}
    # track index
    tdf = defaultdict(set)
    for s in SRCS:
        for i, tk in enumerate(N[s].track_keys):
            if tk:
                for t in tk:
                    if len(t) >= 4: tdf[t].add((s, i))
    cand = defaultdict(set)   # (sa, ia, sb, ib) -> methods
    for sa, sb in combinations(SRCS, 2):
        for meth, X, k in [("name_tfidf", X1, K_NAME), ("name_artist_tfidf", X2, K_NA)]:
            for i, j, c in topk_pairs(X[sa], X[sb], k, MIN_COS):
                cand[(sa, i, sb, j)].add(meth)
            for j, i, c in topk_pairs(X[sb], X[sa], k, MIN_COS):   # symmetric: neighbours from both sides
                cand[(sa, i, sb, j)].add(meth)
        # shared rare tracks
        shared = defaultdict(int)
        for t, recs in tdf.items():
            if len(recs) > 30: continue
            A = [i for s, i in recs if s == sa]; B = [i for s, i in recs if s == sb]
            for i in A:
                for j in B:
                    shared[(i, j)] += 1
        for (i, j), n in shared.items():
            na_, nb_ = N[sa].n_tracks[i], N[sb].n_tracks[j]
            if n >= 2 or min(na_, nb_) <= 2:
                cand[(sa, i, sb, j)].add("tracks")
    rows = [dict(src1=sa, id1=N[sa].id[i], src2=sb, id2=N[sb].id[j], methods="|".join(sorted(m)))
            for (sa, i, sb, j), m in cand.items()]
    C = pd.DataFrame(rows)
    C.to_pickle(f"{W}/state/candidates.pkl")
    os.makedirs(f"{ROOT}/submission/blocking", exist_ok=True)
    C[["id1", "id2"]].to_csv(f"{ROOT}/submission/blocking/candidates.csv", index=False)
    # diagnostics
    diag = {"stage": "blocking", "ts": datetime.datetime.now().isoformat(), "inputs": "work/state/norm_*.pkl",
            "n_candidates": len(C), "by_pair": C.groupby(["src1", "src2"]).size().to_dict()}
    diag["by_pair"] = {f"{a}-{b}": int(v) for (a, b), v in diag["by_pair"].items()}
    full = {f"{a}-{b}": len(N[a]) * len(N[b]) for a, b in combinations(SRCS, 2)}
    diag["reduction_ratio"] = {k: round(1 - diag["by_pair"].get(k, 0) / v, 6) for k, v in full.items()}
    meth = C.methods.str.split("|").explode().value_counts().to_dict()
    diag["by_method"] = {k: int(v) for k, v in meth.items()}
    diag["unique_to_method"] = {k: int((C.methods == k).sum()) for k in meth}
    for s in SRCS:
        ids = pd.concat([C.id1, C.id2])
        cnt = ids.value_counts()
        n = N[s].id.map(cnt).fillna(0)
        diag[f"{s}_zero_candidate_records"] = int((n == 0).sum())
        diag[f"{s}_cand_per_record_p50_p99_max"] = [float(n.median()), float(n.quantile(.99)), float(n.max())]
    print(json.dumps(diag, indent=1))
    with open(f"{W}/diagnostics.jsonl", "a") as f:
        f.write(json.dumps(diag) + "\n")

if __name__ == "__main__":
    main()
