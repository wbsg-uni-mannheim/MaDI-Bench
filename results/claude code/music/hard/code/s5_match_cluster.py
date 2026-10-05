"""Stage 4b+5: rule-based evidence score, one-to-one assignment per source pair, and constrained
clustering (at most one record per source per cluster). Writes work/state/s5_*.pkl"""
import os, json, sys
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = f"{BASE}/work/state"
CFG = dict(threshold=3.0)


def nz(x):
    return x == x  # not NaN


def score_row(r):
    """hand-set additive evidence (log-odds-like points). Missing evidence contributes 0."""
    s = 0.0
    t, ts, tit = r.t_sim, r.t_strict, r.t_in_tracks
    if not nz(t):
        t = ts = 0.0
    if ts >= 0.95:
        s += 2.0
    elif ts >= 0.85:
        s += 1.5
    elif t >= 0.9 and ts >= 0.7:            # word dropped / added (observed noise mode)
        s += 1.0
    elif t >= 0.9 and r.t_prefix == 1 and nz(r.a_sim) and r.a_sim >= 0.9:
        s += 1.0                            # truncated title ('These' / 'These Days') by the same artist
    elif t >= 0.9:                          # mere containment ('Monster' in 'Zenboy vs. Bionic Monster 01')
        s += 0.0
    elif tit >= 0.95:                       # release titled after one of its tracks: neutral, not penalised
        s += 0.0
    elif ts < 0.5:
        s -= 2.0
    else:
        s -= 0.5
    if r.num_conflict == 1:
        s -= 2.0
    a = r.a_sim
    if not nz(a):
        s -= 0.5                            # artist unknown on one side: title evidence alone is weaker
    else:
        if a >= 0.9:
            s += 2.0
        elif a >= 0.75:
            s += 1.0
        elif a < 0.6:
            s -= 2.0
    if nz(r.tr_short):
        big = max(r.n_tr1, r.n_tr2)
        # lastfm track lists are often partial/extended versions of the same release (observed on exact
        # title+artist pairs: 25% have tr_long < 0.65), so list-length disagreement is only used for discogs-musicbrainz
        tl = 1.0 if ("lastfm" in (r.s1, r.s2) and r.tr_short >= 0.8 and r.tr_n >= 2) else r.tr_long
        if tl < 0.3 and big >= 4:       # small overlap with a much longer list: different release
            s -= 1.5
        elif r.tr_short >= 0.8 and tl >= 0.6 and r.tr_n >= 5:
            s += 4.0
        elif r.tr_short >= 0.8 and tl >= 0.5 and r.tr_n >= 2:
            s += 3.0
        elif r.tr_short >= 0.8 and tl >= 0.5:
            s += 1.5
        elif r.tr_short >= 0.5:
            s += 0.5
        elif r.tr_short < 0.2:
            s -= 2.0
        else:
            s -= 0.5
    if nz(r.dur_rel):
        if r.dur_diff <= 10 or r.dur_rel <= 0.02:
            s += 1.5
        elif r.dur_rel <= 0.1:
            s += 0.5
        elif r.dur_rel > 0.3:
            s -= 1.0
    if nz(r.year_diff):
        if r.year_diff == 0:
            s += 0.5
        elif r.year_diff >= 2:
            s -= 1.5
    if nz(r.date_eq) and r.date_eq == 1:
        s += 1.0
    if nz(r.country_eq):
        s += 0.5 if r.country_eq == 1 else -1.0
    return s


def one_to_one(e):
    """greedy max-score one-to-one within a source pair; deterministic tie-break on ids"""
    e = e.sort_values(["fine", "id1", "id2"], ascending=[False, True, True])
    u1, u2, keep = set(), set(), []
    for idx, a, b in zip(e.index, e.id1, e.id2):
        if a in u1 or b in u2:
            continue
        u1.add(a); u2.add(b); keep.append(idx)
    return e.loc[keep]


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else CFG["threshold"]
    df = pd.read_pickle(f"{ST}/s2_all.pkl")
    c = pd.read_pickle(f"{ST}/s4_features.pkl")
    c["score"] = [score_row(r) for r in c.itertuples()]
    # continuous tie-break used only for ranking inside one-to-one assignment (does not change acceptance)
    fz = lambda v: 0.0 if v != v else v
    c["fine"] = c.score + 0.01 * (c.t_strict.map(fz) + c.a_sim.map(fz) + c.tr_long.map(fz)
                                  - c.dur_rel.map(lambda v: 0.5 if v != v else min(v, 1.0)))
    c.to_pickle(f"{ST}/s5_scored.pkl")
    acc = c[c.score >= thr].copy()
    # triangle support: an edge (x,y) whose two ends are both accepted-linked to the same third-source record z
    # gets a small ranking bonus (<0.5, so it never outranks a coarser evidence step)
    nb = {}
    for a, b, s in zip(acc.id1, acc.id2, acc.score):
        nb.setdefault(a, {})[b] = s; nb.setdefault(b, {})[a] = s
    sup = []
    for a, b in zip(acc.id1, acc.id2):
        common = set(nb[a]) & set(nb[b])
        sup.append(max((min(nb[a][z], nb[b][z]) for z in common), default=0.0))
    acc["tri_support"] = sup
    acc["fine"] = acc.fine + 0.04 * acc.tri_support.clip(upper=10)
    # issue ambiguity: when a lastfm/musicbrainz record has >= 3 discogs issues with the identical (maximal) evidence
    # score, picking one is a coin flip (expected precision <= 1/3); such discogs links are withheld.
    top = acc.groupby(["id2", "s1"]).score.transform("max")
    ntie = acc[acc.score == top].groupby(["id2", "s1"]).size()
    key2 = list(zip(acc.id2, acc.s1))
    acc["n_tied"] = [ntie.get(k, 0) if sc == t else 0 for k, sc, t in zip(key2, acc.score, top)]
    ambiguous = (acc.s1 == "discogs") & (acc.n_tied >= 3) & (acc.tri_support == 0)   # a shared third record disambiguates
    n_amb_records = acc[ambiguous].id2.nunique()
    acc = acc[~ambiguous]
    edge_score_all = {(a, b): s for a, b, s in zip(c.id1, c.id2, c.score)}
    # constrained greedy clustering over all accepted edges, strongest first: an edge is rejected if it would put
    # two records of one source into a cluster (entities are single issues; each source lists an issue once),
    # or if any other cross pair of the two clusters has negative evidence (weak-bridge guard).
    parent = {i: i for i in df.id}
    srcs = {i: {s} for i, s in zip(df.id, df.source)}
    members = {i: [i] for i in df.id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    kept = acc.sort_values(["fine", "id1", "id2"], ascending=[False, True, True])
    status = []
    for a, b in zip(kept.id1, kept.id2):
        ra, rb = find(a), find(b)
        if ra == rb:
            status.append("same"); continue
        if srcs[ra] & srcs[rb]:
            status.append("rejected_source_conflict"); continue
        bad = False
        for u in members[ra]:
            for v in members[rb]:
                sc = edge_score_all.get((u, v), edge_score_all.get((v, u)))
                if sc is not None and sc <= 0:
                    bad = True
        if bad:
            status.append("rejected_contradiction"); continue
        parent[rb] = ra; srcs[ra] |= srcs[rb]; members[ra] += members[rb]
        status.append("merged")
    kept["status"] = status
    kept.to_pickle(f"{ST}/s5_edges.pkl")
    roots = {i: find(i) for i in df.id}
    mem = pd.DataFrame({"record_id": df.id, "source": df.source, "root": [roots[i] for i in df.id]})
    # cluster id: smallest member id in a fixed source order (discogs, lastfm, musicbrainz) -> deterministic
    order = {"discogs": 0, "lastfm": 1, "musicbrainz": 2}
    mem["o"] = mem.source.map(order)
    first = mem.sort_values(["o", "record_id"]).groupby("root").record_id.first()
    mem["cluster_id"] = "c_" + mem.root.map(first)
    mem[["record_id", "source", "cluster_id"]].to_pickle(f"{ST}/s5_membership.pkl")
    # correspondences: all cross-source pairs within final clusters (score from edge if direct)
    edge_score = {(a, b): s for a, b, s in zip(c.id1, c.id2, c.score)}
    corr = []
    for cid, g in mem.groupby("cluster_id"):
        ids = list(g.record_id)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                s = edge_score.get((a, b), edge_score.get((b, a), np.nan))
                corr.append((a, b, s))
    corr = pd.DataFrame(corr, columns=["id1", "id2", "score"])
    corr.to_pickle(f"{ST}/s5_corr.pkl")
    sizes = mem.groupby("cluster_id").size()
    combo = mem.groupby("cluster_id").source.apply(lambda s: "+".join(sorted(s)))
    diag = dict(stage="match_cluster", threshold=thr, accepted_edges=int(len(acc)), ambiguous_issue_records=int(n_amb_records),
                edge_status=kept.status.value_counts().to_dict(),
                by_pair={f"{a}-{b}": int(v) for (a, b), v in kept[kept.status == "merged"].groupby(["s1", "s2"]).size().items()},
                n_clusters=int(len(sizes)), size_dist=sizes.value_counts().sort_index().to_dict(),
                source_combos=combo[sizes > 1].value_counts().to_dict(),
                singleton_share=float((sizes == 1).mean()), n_corr=len(corr),
                corr_not_in_candidates=int(sum((a, b) not in edge_score and (b, a) not in edge_score for a, b in zip(corr.id1, corr.id2))))
    print(json.dumps(diag, indent=1))
    json.dump(diag, open(f"{ST}/s5_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
