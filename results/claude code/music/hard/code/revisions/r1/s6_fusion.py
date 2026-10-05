"""Stage 6: attribute-wise fusion from final membership + export of all submission files."""
import os, re, json
import numpy as np, pandas as pd
from normlib import fix_mojibake, is_moji, clean_ws, key, clean_title

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = f"{BASE}/work/state"
SUB = f"{BASE}/submission"
PRIO = {"discogs": 0, "musicbrainz": 1, "lastfm": 2}   # documented deterministic tie-break only
TARGET = ["id", "name", "artist", "release-date", "release-country", "label", "genre", "tracks", "duration"]


def tokset(s):
    return frozenset(key(s).split())


def choose_text(cands, flags_fn, group_key):
    """cands: list of (value, source). Group by order-insensitive token key; pick the group with most members;
    within/between ties prefer values whose tokens are a superset of other members' (truncation/word-drop noise
    only removes tokens), then fewer noise flags, then source priority."""
    cands = [(v, s) for v, s in cands if v]
    if not cands:
        return None, "missing"
    keys = [group_key(v) for v, _ in cands]
    votes = [sum(k == k2 for k2 in keys) for k in keys]
    sup = [sum(1 for k2 in keys if k2 < k) for k in keys]     # strict superset of how many others
    ranked = sorted(range(len(cands)), key=lambda i: (-votes[i], -sup[i], flags_fn(cands[i][0]), PRIO[cands[i][1]], cands[i][0]))
    i = ranked[0]
    how = "unanimous" if votes[i] == len(cands) else ("majority" if votes[i] > 1 else ("superset" if sup[i] else "tie-break"))
    return cands[i][0], how


def title_flags(v):
    f = 0
    f += 2 * ("  " in v)                       # a deleted word leaves a double space
    f += 2 * is_moji(v)
    f += v.count("(") != v.count(")")          # truncated inside parentheses
    f += bool(re.search(r"\b(?:Mus|Orig|Coll)\.", v))   # abbreviated words
    return f


def artist_flags(v):
    f = 0
    f += 2 * bool(re.search(r"(?:^|\s)[A-Z]\.(?:\s|$)", v))   # 'L. Garnier', 'Refugees H.'
    f += 2 * ("(." in v or v.endswith("("))
    f += bool(re.search(r"\bfeat\.?$|\(and others\)", v, re.I))
    f += 2 * is_moji(v)
    f += "," in v and "|" not in v              # 'Garnier, Laurent' inversion
    return f


def artist_key(v):
    return frozenset(key(re.sub(r"\s*\(\d+\)", "", v)).split())


def main():
    df = pd.read_pickle(f"{ST}/s2_all.pkl")
    R = {r.id: r for r in df.itertuples(index=False)}
    mem = pd.read_pickle(f"{ST}/s5_membership.pkl")
    groups = {}
    for rid, cid in zip(mem.record_id, mem.cluster_id):
        groups.setdefault(cid, []).append(rid)
    rows, prov = [], []
    for cid in sorted(groups):
        recs = [R[i] for i in groups[cid]]
        out = {"_id": cid, "id": cid[2:]}
        P = {}
        # name
        out["name"], P["name"] = choose_text([(r.title, r.source) for r in recs], title_flags, tokset)
        # artist: raw display (keeps discogs '(n)' disambiguators, which other sources also carry); lastfm rows
        # without artist fall back to the 'Artist -  Title' prefix
        arts = []
        for r in recs:
            a = clean_ws(fix_mojibake(r.raw_artist)) if r.raw_artist.strip() else (r.title_prefix_artist or "")
            arts.append((a, r.source))
        out["artist"], P["artist"] = choose_text(arts, artist_flags, artist_key)
        # release-date
        ds = [(r.date, r.date_prec, r.source) for r in recs if r.date]
        if ds:
            iso = [d for d, _, _ in ds]
            rank = {"day": 0, "month": 2, "year": 3}
            best = sorted(ds, key=lambda x: (-iso.count(x[0]), rank[x[1]] + (1 if x[1] == "day" and x[0].endswith("-01-01") else 0),
                                              PRIO[x[2]], x[0]))[0]
            out["release-date"] = best[0]; P["release-date"] = "agree" if len(set(iso)) == 1 else "conflict"
        else:
            out["release-date"] = None; P["release-date"] = "missing"
        # release-country
        cs = [(r.country, r.source) for r in recs if r.country]
        if cs:
            vals = [c for c, _ in cs]
            best = sorted(cs, key=lambda x: (-vals.count(x[0]), PRIO[x[1]]))[0]
            out["release-country"] = best[0]; P["release-country"] = "agree" if len(set(vals)) == 1 else "conflict"
        else:
            out["release-country"] = None; P["release-country"] = "missing"
        # duration: a value confirmed by another member within 5 s wins; otherwise source priority
        du = [(int(r.duration_s), r.source) for r in recs if r.duration_s == r.duration_s and r.duration_s]
        if du:
            sup = [sum(abs(d - d2) <= 5 for d2, _ in du) for d, _ in du]
            best = sorted(range(len(du)), key=lambda i: (-sup[i], PRIO[du[i][1]], du[i][0]))[0]
            out["duration"] = du[best][0]; P["duration"] = "agree" if min(sup) == len(du) else ("majority" if sup[best] > 1 else "tie-break")
        else:
            out["duration"] = None; P["duration"] = "missing"
        # tracks
        tl = [(r.tracks, r.source) for r in recs if r.tracks]
        if tl:
            ks = [frozenset(t.lower() for t in l) for l, _ in tl]
            votes = [sum(k == k2 for k2 in ks) for k in ks]
            fl = [sum(is_moji(t) for t in l) for l, _ in tl]
            best = sorted(range(len(tl)), key=lambda i: (-votes[i], fl[i], PRIO[tl[i][1]]))[0]
            out["tracks"] = json.dumps(tl[best][0], ensure_ascii=False)
            P["tracks"] = "agree" if min(votes) == len(tl) else ("majority" if votes[best] > 1 else "tie-break")
        else:
            out["tracks"] = None; P["tracks"] = "missing"
        # label / genre: discogs only
        d = [r for r in recs if r.source == "discogs"]
        if d:
            r = d[0]
            out["label"] = "|".join(r.labels) if r.labels else None
            out["genre"] = r.genre if r.genre else None
        else:
            out["label"] = out["genre"] = None
        rows.append(out)
        prov.append({"_id": cid, "n_members": len(recs), **{f"how_{k}": v for k, v in P.items()}})
    fused = pd.DataFrame(rows)[["_id"] + TARGET]
    fused["duration"] = fused["duration"].astype("Int64")
    os.makedirs(SUB, exist_ok=True)
    fused.to_csv(f"{SUB}/fused.csv", index=False)
    pd.DataFrame(prov).to_csv(f"{ST}/s6_provenance.csv", index=False)
    # membership / correspondences / candidates
    mem[["record_id", "source", "cluster_id"]].to_csv(f"{SUB}/membership.csv", index=False)
    corr = pd.read_pickle(f"{ST}/s5_corr.pkl")
    corr["score"] = corr.score.fillna(0).map(lambda s: round(min(max(s / 10, 0.3), 1.0), 3))
    corr.to_csv(f"{SUB}/correspondences.csv", index=False)
    cand = pd.read_pickle(f"{ST}/s3_candidates.pkl")[["id1", "id2"]]
    cand = pd.concat([cand, corr[["id1", "id2"]]])
    cand = cand[~pd.Series([tuple(sorted(p)) for p in zip(cand.id1, cand.id2)]).duplicated().values]
    os.makedirs(f"{SUB}/blocking", exist_ok=True)
    cand.to_csv(f"{SUB}/blocking/candidates.csv", index=False)
    print("fused", fused.shape, "density:", fused.notna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
