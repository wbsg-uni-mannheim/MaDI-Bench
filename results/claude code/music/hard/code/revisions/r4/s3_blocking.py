"""Stage 3: blocking. Union of complementary candidate generators across the three source pairs:
  A) title char-3gram TF-IDF top-k neighbours
  B) track-list word TF-IDF top-k neighbours (records with tracks only)
  C) artist+title combined token TF-IDF top-k neighbours
Writes work/state/s3_candidates.pkl (idx1, idx2, methods) and diagnostics."""
import os, json, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = f"{BASE}/work/state"
K = {"title": 10, "tracks": 10, "combo": 10}
MIN_SIM = {"title": 0.35, "tracks": 0.25, "combo": 0.3}
PAIRS = [("discogs", "lastfm"), ("discogs", "musicbrainz"), ("lastfm", "musicbrainz")]


def topk(A, B, k, min_sim, chunk=2000):
    """for each row of A: top-k rows of B by cosine (A,B L2-normalised). returns (i, j, s)."""
    out_i, out_j, out_s = [], [], []
    BT = B.T.tocsr()
    for st in range(0, A.shape[0], chunk):
        S = (A[st:st + chunk] @ BT).toarray()
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        sc = np.take_along_axis(S, idx, axis=1)
        r = np.repeat(np.arange(S.shape[0]), kk)
        m = sc.ravel() >= min_sim
        out_i.append(r[m] + st); out_j.append(idx.ravel()[m]); out_s.append(sc.ravel()[m])
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)


def main():
    t0 = time.time()
    df = pd.read_pickle(f"{ST}/s2_all.pkl")
    texts = {
        "title": df["k_title"].fillna(""),
        "tracks": df["k_tracks"].map(lambda l: " ".join(l)),
        "combo": (df["k_artist"].fillna("") + " " + df["k_title"].fillna("")).str.strip(),
    }
    vecs = {
        "title": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True),
        "tracks": TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]{2,}", min_df=1, max_df=0.05, sublinear_tf=True),
        "combo": TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=1, sublinear_tf=True),
    }
    cands = {}
    stats = []
    for m, txt in texts.items():
        X = vecs[m].fit_transform(txt)
        for s1, s2 in PAIRS:
            i1 = np.where((df.source == s1).values & (txt.str.len() > 0).values)[0]
            i2 = np.where((df.source == s2).values & (txt.str.len() > 0).values)[0]
            n0 = len(cands)
            for (ia, ib, flip) in [(i1, i2, False), (i2, i1, True)]:
                a, b, s = topk(X[ia], X[ib], K[m], MIN_SIM[m])
                for x, y, v in zip(ia[a], ib[b], s):
                    p = (y, x) if flip else (x, y)
                    d = cands.setdefault(p, {})
                    d[m] = max(d.get(m, 0), float(v))
            stats.append(dict(method=m, pair=f"{s1}-{s2}", n_query_a=len(i1), n_query_b=len(i2), new_pairs=len(cands) - n0))
            print(m, s1, s2, len(cands) - n0, f"{time.time()-t0:.0f}s", flush=True)
    # D) exact title-key blocks: all cross-source pairs sharing the normalised title (keeps every issue of a
    #    title reachable, which top-k cannot guarantee for 15+ issues); blocks with >60 records per source are
    #    refined by adding each artist token as a sub-key.
    n0 = len(cands)
    src = df.source.values
    groups = {}
    for i, (t, a) in enumerate(zip(df.k_title.fillna(""), df.k_artist.fillna(""))):
        if t:
            groups.setdefault(t, []).append(i)
    for t, idx in groups.items():
        by = {}
        for i in idx:
            by.setdefault(src[i], []).append(i)
        if len(by) < 2:
            continue
        big = max(len(v) for v in by.values()) > 60
        for s1, s2 in PAIRS:
            for x in by.get(s1, []):
                ax = set(df.k_artist.values[x].split())
                for y in by.get(s2, []):
                    if big and not (ax & set(df.k_artist.values[y].split())):
                        continue
                    d = cands.setdefault((x, y), {})
                    d["exact"] = 1.0
    stats.append(dict(method="exact_title", new_pairs=len(cands) - n0))
    print("exact", len(cands) - n0)
    rows = [(a, b, d.get("title", np.nan), d.get("tracks", np.nan), d.get("combo", np.nan)) for (a, b), d in cands.items()]
    c = pd.DataFrame(rows, columns=["i1", "i2", "b_title", "b_tracks", "b_combo"])
    c["b_exact"] = [cands[(a, b)].get("exact", np.nan) for a, b in zip(c.i1, c.i2)]
    c["id1"] = df.id.values[c.i1]; c["id2"] = df.id.values[c.i2]
    c["s1"] = df.source.values[c.i1]; c["s2"] = df.source.values[c.i2]
    c.to_pickle(f"{ST}/s3_candidates.pkl")
    # diagnostics
    n = df.source.value_counts().to_dict()
    total = n["discogs"] * n["lastfm"] + n["discogs"] * n["musicbrainz"] + n["lastfm"] * n["musicbrainz"]
    per_rec = pd.concat([c.id1, c.id2]).value_counts()
    zero = {s: int((~df[df.source == s].id.isin(per_rec.index)).sum()) for s in n}
    diag = dict(stage="blocking", n_candidates=len(c), full_cross=total, reduction_ratio=1 - len(c) / total,
                by_pair=c.groupby(["s1", "s2"]).size().rename(lambda x: f"{x[0]}-{x[1]}").to_dict() if False else
                {f"{a}-{b}": int(v) for (a, b), v in c.groupby(["s1", "s2"]).size().items()},
                only_title=int((c.b_title.notna() & c.b_tracks.isna() & c.b_combo.isna()).sum()),
                only_tracks=int((c.b_tracks.notna() & c.b_title.isna() & c.b_combo.isna()).sum()),
                only_combo=int((c.b_combo.notna() & c.b_title.isna() & c.b_tracks.isna()).sum()),
                zero_candidate_records=zero, per_record_max=int(per_rec.max()), per_record_median=float(per_rec.median()),
                method_stats=stats)
    print(json.dumps(diag, indent=1))
    json.dump(diag, open(f"{ST}/s3_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
