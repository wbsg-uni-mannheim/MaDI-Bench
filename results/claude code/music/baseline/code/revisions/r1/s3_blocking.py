"""Stage 3: blocking. Union of TF-IDF nearest neighbours (both directions, each source pair):
 A) title char 3-5grams (top K_NAME)  B) title+artist word tokens (top K_TA)  C) track-list words (top K_TR)."""
import pandas as pd, numpy as np, json, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
K_NAME, K_TA, K_TR = 10, 5, 5
R = pd.read_pickle("work/state/records.pkl")
srcs = ["discogs", "lastfm", "musicbrainz"]
def topk(A, B, k):
    out = []
    for s in range(0, A.shape[0], 2000):
        S = (A[s:s+2000] @ B.T).toarray()
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk-1, axis=1)[:, :kk]
        for i in range(S.shape[0]):
            for j in idx[i]:
                if S[i, j] > 0: out.append((s+i, j, S[i, j]))
    return out
views = {
 "name": (lambda r: r.k_name, dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True), K_NAME),
 "title_artist": (lambda r: r.k_name + " " + " ".join(t for t in r.k_artist.split() if len(t) > 1),
                  dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True), K_TA),
 "tracks": (lambda r: " ".join(r.k_tracks), dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, min_df=1), K_TR),
}
pairs = {}
for vname, (f, kw, k) in views.items():
    txt = R.apply(f, axis=1)
    V = TfidfVectorizer(**kw).fit(txt[txt.str.len() > 0])
    X = V.transform(txt)
    for a, b in itertools.combinations(srcs, 2):
        ia = np.where((R.source == a) & (txt.str.len() > 0))[0]; ib = np.where((R.source == b) & (txt.str.len() > 0))[0]
        for (i, j, s) in topk(X[ia], X[ib], k) + [(j, i, s) for (i, j, s) in topk(X[ib], X[ia], k)]:
            p = (R.id.iat[ia[i]], R.id.iat[ib[j]])
            pairs.setdefault(p, set()).add(vname)
C = pd.DataFrame([(a, b, "|".join(sorted(v))) for (a, b), v in pairs.items()], columns=["id1", "id2", "views"])
C.to_csv("work/state/candidates.csv", index=False)
C[["id1", "id2"]].to_csv("submission/blocking/candidates.csv", index=False)
src = dict(zip(R.id, R.source))
C["sp"] = C.id1.map(src) + "-" + C.id2.map(src)
n = R.source.value_counts()
full = n["discogs"]*n["lastfm"] + n["discogs"]*n["musicbrainz"] + n["lastfm"]*n["musicbrainz"]
cnt = pd.concat([C.id1, C.id2]).value_counts()
diag = dict(stage="blocking", candidates=len(C), by_pair=C.sp.value_counts().to_dict(),
            reduction_ratio=1 - len(C)/full, views=C.views.value_counts().to_dict(),
            zero_candidate_records=int((~R.id.isin(cnt.index)).sum()),
            per_record_max=int(cnt.max()), per_record_median=float(cnt.median()))
print(json.dumps(diag, indent=1))
import datetime
diag.update(ts=datetime.datetime.now().isoformat(), inputs="work/state/records.pkl", revision="r1")
open("work/diagnostics.jsonl", "a").write(json.dumps(diag) + "\n")
