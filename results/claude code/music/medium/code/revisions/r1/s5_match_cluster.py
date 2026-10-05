"""Stage 5: direct rule-based pair scoring + constrained greedy clustering.
Score = sum of interpretable evidence points (hand-set, documented in report). No training.
Clustering: accepted edges sorted by score; union only if the merged cluster keeps <=1 record per
source (one row per release per source; checked in profiling: musicbrainz has ~0 internal duplicates,
discogs duplicates are different issues = different entities). Edges whose record has >=3 near-tied
alternatives in the other source are 'ambiguous' and only used if the ambiguity disappears after the
unambiguous edges have been placed."""
import pandas as pd, numpy as np, os, json, datetime, sys
from collections import defaultdict

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["discogs", "lastfm", "musicbrainz"]
CFG = dict(T_ACCEPT=4.5, TIE_EPS=0.5, MAX_TIES=2)
if len(sys.argv) > 1:
    CFG.update(json.loads(sys.argv[1]))

def pts_name(s):
    return 3.0 if s >= 0.97 else 2.0 if s >= 0.9 else 0.5 if s >= 0.8 else -1.0 if s >= 0.7 else -3.0

def pts_artist(s):
    if np.isnan(s): return 0.0
    return 2.0 if s >= 0.9 else 1.0 if s >= 0.7 else -0.5 if s >= 0.5 else -3.0

def pts_tracks(ts, tc, n1, n2):
    if np.isnan(ts): return 0.0
    if ts >= 0.9: return 3.0
    if ts >= 0.7: return 2.0
    if tc >= 0.8: return 1.0
    if tc >= 0.5: return 0.0
    return -2.0 if min(n1, n2) >= 2 else -1.0

def pts_duration(d1, d2, has_lastfm):
    if np.isnan(d1) or np.isnan(d2): return 0.0
    d = abs(d1 - d2)
    if has_lastfm:  # lastfm durations/tracklists are often partial -> only agreement counts
        return 1.5 if d <= 5 else 0.5 if d <= 30 else 0.0
    if d <= 10: return 2.0
    if d <= 60: return 1.0
    if d <= 300 or d / max(d1, d2) <= 0.1: return 0.0
    return -2.0

def pts_date(dt1, dt2, y1, y2):
    if isinstance(dt1, str) and isinstance(dt2, str) and dt1 == dt2:
        return 1.0 if dt1.endswith("-01-01") else 2.0
    if np.isnan(y1) or np.isnan(y2): return 0.0
    dy = abs(y1 - y2)
    return 0.5 if dy == 0 else -0.5 if dy == 1 else -2.0

def pts_country(c1, c2):
    if not isinstance(c1, str) or not isinstance(c2, str): return 0.0
    return 1.0 if c1 == c2 else -0.5

def score(F):
    lf = (F.src1 == "lastfm") | (F.src2 == "lastfm")
    parts = pd.DataFrame(index=F.index)
    parts["p_name"] = F.name_sim.map(pts_name)
    parts["p_artist"] = F.artist_sim.map(pts_artist)
    parts["p_tracks"] = [pts_tracks(a, b, c, d) for a, b, c, d in zip(F.track_sim, F.track_cont, F.ntr1, F.ntr2)]
    parts["p_ntracks"] = np.where((F.ntr1 == F.ntr2) & (F.ntr1 >= 3) & ~lf, 0.5, 0.0)
    parts["p_dur"] = [pts_duration(a, b, l) for a, b, l in zip(F.dur1, F.dur2, lf)]
    parts["p_date"] = [pts_date(a, b, c, d) for a, b, c, d in zip(F.date1, F.date2, F.year1.astype(float), F.year2.astype(float))]
    parts["p_ctry"] = [pts_country(a, b) for a, b in zip(F.ctry1, F.ctry2)]
    return parts

class UF:
    def __init__(self, recs):
        self.p = {r: r for r in recs}; self.members = {r: {r} for r in recs}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def srcs(self, r):
        return {m.split("_")[0] for m in self.members[self.find(r)]}
    def can(self, a, b):
        ra, rb = self.find(a), self.find(b)
        return ra != rb and not (self.srcs(ra) & self.srcs(rb))
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        self.p[rb] = ra; self.members[ra] |= self.members.pop(rb)

def main():
    F = pd.read_pickle(f"{W}/state/features.pkl")
    P = score(F)
    F = pd.concat([F, P], axis=1)
    F["score"] = P.sum(axis=1)
    F.to_pickle(f"{W}/state/scored.pkl")
    A = F[F.score >= CFG["T_ACCEPT"]].copy()
    # ambiguity: number of near-tied alternatives of each endpoint within this source pair
    def ties(col_self, col_other):
        best = A.groupby([col_self, "src1", "src2"]).score.transform("max")
        near = A.score >= best - CFG["TIE_EPS"]
        cnt = A[near].groupby([col_self, "src1", "src2"]).size()
        return near, A.set_index([col_self, "src1", "src2"]).index.map(cnt).values
    near1, n1 = ties("id1", "id2"); near2, n2 = ties("id2", "id1")
    A["is_best"] = near1.values & near2.values
    A["ties"] = np.maximum(n1, n2)
    A = A[A.is_best]   # an edge must be (near-)best for both endpoints
    A["ambiguous"] = A.ties > CFG["MAX_TIES"]
    A = A.sort_values(["score", "id1", "id2"], ascending=[False, True, True])
    recs = pd.concat([pd.read_pickle(f"{W}/state/norm_{s}.pkl").id for s in SRCS])
    # record id prefixes: discogs_, lastFM_, mbrainz_ -> source label via map
    pref = {"discogs": "discogs", "lastFM": "lastfm", "mbrainz": "musicbrainz"}
    uf = UF(recs)
    uf.srcs = lambda r: {pref[m.split("_")[0]] for m in uf.members[uf.find(r)]}
    accepted, rejected = [], []
    for r in A[~A.ambiguous].itertuples():
        if uf.can(r.id1, r.id2):
            uf.union(r.id1, r.id2); accepted.append((r.id1, r.id2, r.score, "unambiguous"))
        elif uf.find(r.id1) != uf.find(r.id2):
            rejected.append((r.id1, r.id2, r.score, "source_conflict"))
    # second pass: ambiguous edges, only if the endpoint has <= MAX_TIES still-available tied options
    amb = A[A.ambiguous]
    for key, g in amb.groupby(["id1", "src2"]):
        pass
    avail_cnt = {}
    for r in amb.itertuples():
        for me, other_src in ((r.id1, r.src2), (r.id2, r.src1)):
            k = (me, other_src)
            if k not in avail_cnt:
                col_me = "id1" if me == r.id1 else "id2"; col_o = "id2" if col_me == "id1" else "id1"
                opts = amb[amb[col_me] == me][col_o]
                avail_cnt[k] = sum(uf.can(me, o) for o in opts)
    for r in amb.itertuples():
        if avail_cnt[(r.id1, r.src2)] <= CFG["MAX_TIES"] and avail_cnt[(r.id2, r.src1)] <= CFG["MAX_TIES"] and uf.can(r.id1, r.id2):
            uf.union(r.id1, r.id2); accepted.append((r.id1, r.id2, r.score, "ambiguous_resolved"))
        else:
            rejected.append((r.id1, r.id2, r.score, "ambiguous"))
    # clusters
    rows = []
    for root, mem in uf.members.items():
        mem = sorted(mem)
        # cluster id: deterministic, based on the smallest member id per preferred source order
        cid = "c_" + sorted(mem, key=lambda x: (["discogs", "lastFM", "mbrainz"].index(x.split("_")[0]), x))[0]
        for m in mem:
            rows.append(dict(record_id=m, source=pref[m.split("_")[0]], cluster_id=cid))
    M = pd.DataFrame(rows)
    M.to_pickle(f"{W}/state/membership.pkl")
    pd.DataFrame(accepted, columns=["id1", "id2", "score", "pass"]).to_pickle(f"{W}/state/accepted_edges.pkl")
    pd.DataFrame(rejected, columns=["id1", "id2", "score", "reason"]).to_pickle(f"{W}/state/rejected_edges.pkl")
    # correspondences = all cross-source pairs within final clusters (consistent with clustering)
    cid = M.set_index("record_id").cluster_id
    corr = []
    sc = F.set_index(["id1", "id2"]).score
    for c, g in M.groupby("cluster_id"):
        ids = sorted(g.record_id)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                s = sc.get((a, b), sc.get((b, a), np.nan))
                corr.append((a, b, s))
    Cr = pd.DataFrame(corr, columns=["id1", "id2", "raw_score"])
    Cr["score"] = (Cr.raw_score.fillna(CFG["T_ACCEPT"]) / 16).clip(0, 1).round(4)  # max attainable ~16 points
    Cr.to_pickle(f"{W}/state/correspondences.pkl")
    sizes = M.groupby("cluster_id").size()
    nsrc = M.groupby("cluster_id").source.nunique()
    comp = M.groupby("cluster_id").source.apply(lambda s: "+".join(sorted(s)))
    diag = {"stage": "match_cluster", "ts": datetime.datetime.now().isoformat(), "cfg": CFG,
            "inputs": "work/state/features.pkl", "n_scored": len(F), "n_above_T": int((F.score >= CFG["T_ACCEPT"]).sum()),
            "n_mutual_best": len(A), "n_ambiguous": int(A.ambiguous.sum()),
            "accepted_edges": len(accepted), "rejected_edges": len(rejected),
            "rejected_by_reason": pd.Series([r[3] for r in rejected]).value_counts().to_dict() if rejected else {},
            "n_records": len(M), "n_clusters": int(sizes.size), "size_dist": sizes.value_counts().sort_index().to_dict(),
            "singleton_share": round(float((sizes == 1).mean()), 4), "composition": comp.value_counts().to_dict(),
            "n_correspondences": len(Cr), "corr_by_pair": Cr.apply(lambda r: "-".join(sorted([r.id1.split('_')[0], r.id2.split('_')[0]])), axis=1).value_counts().to_dict()}
    for s in SRCS:
        ms = M[M.source == s]
        diag[f"{s}_matched_share"] = round(float((ms.cluster_id.map(sizes) > 1).mean()), 4)
    diag = json.loads(json.dumps(diag, default=int))
    print(json.dumps(diag, indent=1))
    with open(f"{W}/diagnostics.jsonl", "a") as f:
        f.write(json.dumps(diag) + "\n")

if __name__ == "__main__":
    main()
