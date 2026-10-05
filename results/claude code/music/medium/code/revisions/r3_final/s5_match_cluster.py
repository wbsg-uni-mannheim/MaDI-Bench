"""Stage 5: direct rule-based pair scoring + constrained greedy clustering.
Score = sum of interpretable evidence points (hand-set, documented in report). No training.
Within-source copies: discogs and lastfm contain noisy duplicate copies of the same release (identical
tracklists with dropped tracks / blanked fields). They are first grouped by strict copy rules (copy_edges);
musicbrainz shows no internal copies. Clustering then works on copy-groups: accepted cross-source edges
sorted by score; union only if the merged cluster keeps <=1 copy-group per source (other same-title
discogs records are different issues = different entities). Edges whose record has >=3 near-tied
alternatives in the other source are 'ambiguous' and only used if the ambiguity disappears after the
unambiguous edges have been placed."""
import pandas as pd, numpy as np, os, json, datetime, sys
from collections import defaultdict

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["discogs", "lastfm", "musicbrainz"]
CFG = dict(T_ACCEPT=4.5, T_ARTIST=4.0, TIE_EPS=0.5, MAX_TIES=2, COPY_MIN_POS=2)
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
    return -1.0   # discogs durations are sometimes ~2x the musicbrainz value for otherwise identical releases

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

def _eq(a, b):
    return np.where(pd.notna(a) & pd.notna(b), a == b, np.nan)

def copy_edges():
    """strict within-source duplicate rules (see report)"""
    F = pd.read_pickle(f"{W}/state/features_within.pkl")
    D = F[F.src1 == "discogs"].copy()
    ev = pd.DataFrame({"date": _eq(D.date1, D.date2), "year": _eq(D.year1, D.year2), "ctry": _eq(D.ctry1, D.ctry2),
                       "label": D.label_ov.values,
                       "dur": np.where(D.dur1.notna() & D.dur2.notna(), (D.dur1 - D.dur2).abs() <= 5, np.nan),
                       "trk": np.where(D.track_sim.notna(), (D.track_sim >= 0.8) & ((D.ntr1 - D.ntr2).abs() <= 1), np.nan)},
                      index=D.index).astype(float)
    conflict = (ev == 0).any(axis=1)
    pos = (ev[["date", "ctry", "label", "dur", "trk"]] == 1).sum(axis=1)
    dk = D[(D.name_sim >= 0.9) & ((D.artist_sim >= 0.7) | D.artist_sim.isna()) & ~conflict & (pos >= CFG["COPY_MIN_POS"])]
    L = F[F.src1 == "lastfm"]
    both_tr = L.track_cont.notna()
    art_ok = L.artist_sim >= 0.9
    art_missing_tr = L.artist_sim.isna() & (L.track_cont >= 0.8) & (np.minimum(L.ntr1, L.ntr2) >= 3)
    name_ok = (L.name_sim >= 0.97) | ((L.name_sim >= 0.9) & (L.track_cont >= 0.8))
    lk = L[name_ok & (art_ok | art_missing_tr) & (~both_tr | (L.track_cont >= 0.8))]
    E = pd.concat([dk, lk])[["id1", "id2", "src1"]]
    return E, D.assign(conflict=conflict)

def build_groups(recs, E, Dconf):
    """union copy edges; discogs merges are complete-link: no conflicting pair inside a group"""
    conflict_pairs = set(map(tuple, Dconf[Dconf.conflict][["id1", "id2"]].values))
    conflict_pairs |= {(b, a) for a, b in conflict_pairs}
    g = {r: r for r in recs}; mem = {r: [r] for r in recs}
    def find(x):
        while g[x] != x:
            g[x] = g[g[x]]; x = g[x]
        return x
    for a, b in E[["id1", "id2"]].values:
        ra, rb = find(a), find(b)
        if ra == rb: continue
        if any((x, y) in conflict_pairs for x in mem[ra] for y in mem[rb]): continue
        g[rb] = ra; mem[ra] += mem.pop(rb)
    return {r: find(r) for r in recs}, mem

def main():
    F = pd.read_pickle(f"{W}/state/features.pkl")
    P = score(F)
    F = pd.concat([F, P], axis=1)
    F["score"] = P.sum(axis=1)
    F.to_pickle(f"{W}/state/scored.pkl")
    # accept: score >= T_ACCEPT, or score >= T_ARTIST when the artist positively agrees
    # (typical case: one title word dropped by noise + matching artist, no other attributes available)
    recs_all = pd.concat([pd.read_pickle(f"{W}/state/norm_{s}.pkl").id for s in SRCS]).tolist()
    E, Dconf = copy_edges()
    gof, gmem = build_groups(recs_all, E, Dconf)
    pd.DataFrame([(r, gof[r]) for r in recs_all], columns=["record_id", "group"]).to_pickle(f"{W}/state/copy_groups.pkl")
    A = F[(F.score >= CFG["T_ACCEPT"]) | ((F.score >= CFG["T_ARTIST"]) & (F.p_artist >= 1.0))].copy()
    # lift record edges to copy-group edges (best member pair represents the group pair)
    A["rid1"], A["rid2"] = A.id1, A.id2
    A["id1"], A["id2"] = A.rid1.map(gof), A.rid2.map(gof)
    A = A.sort_values(["score", "rid1", "rid2"], ascending=[False, True, True]).drop_duplicates(["id1", "id2"])
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
    recs = sorted(gmem)
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
    for root, gm in uf.members.items():
        mem = sorted(r for gr in gm for r in gmem[gr])
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
    gsc = A.set_index(["id1", "id2"]).score
    for c, g in M.groupby("cluster_id"):
        ids = sorted(g.record_id)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a.split("_")[0] == b.split("_")[0]:
                    continue   # within-source copies are not cross-source correspondences
                s = sc.get((a, b), sc.get((b, a), np.nan))
                if pd.isna(s):
                    s = gsc.get((gof[a], gof[b]), gsc.get((gof[b], gof[a]), np.nan))
                corr.append((a, b, s))
    Cr = pd.DataFrame(corr, columns=["id1", "id2", "raw_score"])
    Cr["score"] = (Cr.raw_score.fillna(CFG["T_ACCEPT"]) / 16).clip(0, 1).round(4)  # max attainable ~16 points
    Cr.to_pickle(f"{W}/state/correspondences.pkl")
    # copy-group expansion of the blocking set: a record whose copy was a candidate of b is itself a
    # candidate of b; only the expanded pairs that end up asserted need to be added for containment
    C0 = pd.read_pickle(f"{W}/state/candidates.pkl")
    have = set(zip(C0.id1, C0.id2)) | set(zip(C0.id2, C0.id1))
    extra = [(a, b) for a, b in zip(Cr.id1, Cr.id2) if (a, b) not in have]
    Cx = pd.concat([C0[["id1", "id2"]], pd.DataFrame(extra, columns=["id1", "id2"])], ignore_index=True)
    os.makedirs(f"{ROOT}/submission/blocking", exist_ok=True)
    Cx.to_csv(f"{ROOT}/submission/blocking/candidates.csv", index=False)
    sizes = M.groupby("cluster_id").size()
    nsrc = M.groupby("cluster_id").source.nunique()
    comp = M.groupby("cluster_id").source.apply(lambda s: "+".join(sorted(s)))
    diag = {"stage": "match_cluster", "ts": datetime.datetime.now().isoformat(), "cfg": CFG,
            "inputs": "work/state/features.pkl", "n_scored": len(F), "n_above_T": int((F.score >= CFG["T_ACCEPT"]).sum()),
            "n_mutual_best": len(A), "n_ambiguous": int(A.ambiguous.sum()),
            "accepted_edges": len(accepted), "rejected_edges": len(rejected),
            "rejected_by_reason": pd.Series([r[3] for r in rejected]).value_counts().to_dict() if rejected else {},
            "copy_edges": {k: int(v) for k, v in E.src1.value_counts().items()},
            "copy_group_sizes": {f"{k}": int(v) for k, v in pd.Series([len(v) for v in gmem.values()]).value_counts().sort_index().items()},
            "candidates_added_by_copy_expansion": len(extra),
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
