"""Stage 6: fusion of the final membership into one row per cluster, plus submission export.
Resolvers (all values come from the cluster's own normalized member records):
 name     : vote on normalized key; ties -> no word-drop marker, more tokens (noise only drops words), source priority
 artist   : vote on sorted-token key; ties -> fewer abbreviated initials, more tokens, source priority
 date     : full ISO dates only; vote; ties -> not a '-01-01' year placeholder, musicbrainz > discogs
 country  : vote on comparison key; ties -> musicbrainz > discogs
 duration : pairwise agreement within 5 s wins; else musicbrainz > discogs > lastfm
 tracks   : medoid list (highest mean similarity to other members' lists); ties -> discogs > musicbrainz > lastfm
 label    : discogs only; union over discogs copies in the cluster (set-valued attribute)
 genre    : discogs only; vote over discogs copies, ties -> more genres, alphabetical
"""
import pandas as pd, numpy as np, os, json, datetime, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from rapidfuzz import fuzz

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["discogs", "lastfm", "musicbrainz"]
PRI_TEXT = {"discogs": 0, "musicbrainz": 1, "lastfm": 2}
PRI_MB = {"musicbrainz": 0, "discogs": 1, "lastfm": 2}
COLS = ["id", "name", "artist", "release-date", "release-country", "label", "genre", "tracks", "duration"]

def n_initials(a):
    return sum(1 for t in a.replace(".", ". ").split() if len(t.strip(".")) == 1)

def pick_name(recs):
    c = [r for r in recs if isinstance(r["name"], str) and r["name"]]
    if not c: return None, "none"
    votes = Counter(r["name_key"] for r in c)
    best = sorted(c, key=lambda r: (-votes[r["name_key"]], bool(r["name_dropmark"]), -len(r["name_key"].split()),
                                    PRI_TEXT[r["source"]], r["id"]))[0]
    return best["name"], "vote" if votes[best["name_key"]] > 1 else "priority"

def pick_artist(recs):
    c = [r for r in recs if isinstance(r["artist"], str) and r["artist"]]
    if not c: return None, "none"
    key = lambda r: " ".join(sorted(r["artist_key"].split()))
    votes = Counter(key(r) for r in c)
    best = sorted(c, key=lambda r: (-votes[key(r)], n_initials(r["artist"]), -len(r["artist_key"].split()),
                                    PRI_TEXT[r["source"]], r["id"]))[0]
    return best["artist"], "vote" if votes[key(best)] > 1 else "priority"

def pick_date(recs):
    c = [r for r in recs if isinstance(r["date"], str)]
    if not c: return None, "none"
    votes = Counter(r["date"] for r in c)
    best = sorted(c, key=lambda r: (-votes[r["date"]], r["date"].endswith("-01-01"), PRI_MB[r["source"]], r["id"]))[0]
    return best["date"], "vote" if votes[best["date"]] > 1 else "priority"

def pick_country(recs):
    c = [r for r in recs if isinstance(r["country"], str)]
    if not c: return None, "none"
    votes = Counter(r["country_key"] for r in c)
    best = sorted(c, key=lambda r: (-votes[r["country_key"]], PRI_MB[r["source"]], r["id"]))[0]
    return best["country"], "vote" if votes[best["country_key"]] > 1 else "priority"

def pick_duration(recs):
    c = [r for r in recs if r["duration"] is not None and not pd.isna(r["duration"])]
    if not c: return None, "none"
    c = sorted(c, key=lambda r: (PRI_MB[r["source"]], r["id"]))
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if abs(c[i]["duration"] - c[j]["duration"]) <= 5:
                return int(c[i]["duration"]), "agree"
    return int(c[0]["duration"]), "priority"

def pick_tracks(recs):
    """medoid tracklist as skeleton; each skeleton track is replaced by the majority spelling among the
    aligned tracks (fuzzy ratio >= 80) of the other members; ties -> more words (noise drops words),
    no double space, source priority."""
    c = [r for r in recs if r["tracks"]]
    if not c: return None, "none"
    if len(c) == 1: return c[0]["tracks"], "single"
    def sim(a, b):
        A, B = a["track_keys"], b["track_keys"]
        return sum(max(fuzz.ratio(x, y) for y in B) for x in A) / 100 / max(len(A), len(B))
    sc = {r["id"]: np.mean([sim(r, o) for o in c if o is not r]) for r in c}
    c = sorted(c, key=lambda r: (-round(sc[r["id"]], 6), PRI_TEXT[r["source"]], r["id"]))
    skel, others = c[0], c[1:]
    if len(c) < 3:
        return skel["tracks"], "medoid"
    out, changed = [], 0
    from s2_normalize import text_key
    for t in skel["tracks"]:
        k = text_key(t)
        opts = [(t, skel["source"])]
        for o in others:
            best = max(o["tracks"], key=lambda u: fuzz.ratio(k, text_key(u)))
            if fuzz.ratio(k, text_key(best)) >= 80:
                opts.append((best, o["source"]))
        votes = Counter(text_key(u) for u, _ in opts)
        pick = sorted(opts, key=lambda x: (-votes[text_key(x[0])], "  " in x[0], -len(text_key(x[0]).split()),
                                           PRI_TEXT[x[1]]))[0][0]
        changed += text_key(pick) != k
        out.append(pick)
    return out, "medoid_voted" if changed else "medoid"

def main():
    N = pd.concat([pd.read_pickle(f"{W}/state/norm_{s}.pkl") for s in SRCS], ignore_index=True)
    N = N.where(pd.notna(N), None).set_index("id", drop=False)
    M = pd.read_pickle(f"{W}/state/membership.pkl")
    rows, prov = [], []
    for cid, g in M.groupby("cluster_id", sort=True):
        recs = [N.loc[i].to_dict() for i in sorted(g.record_id)]
        rep = cid[2:]
        row = {"_id": cid, "id": rep}
        for attr, fn in [("name", pick_name), ("artist", pick_artist), ("release-date", pick_date),
                         ("release-country", pick_country), ("duration", pick_duration), ("tracks", pick_tracks)]:
            v, how = fn(recs); row[attr] = v; prov.append((cid, attr, how))
        d = [r for r in recs if r["source"] == "discogs"]
        labs = []
        for r in d:                       # union over discogs copies (label is a set attribute)
            for l in (r["label"] or []):
                if l not in labs: labs.append(l)
        row["label"] = "|".join(labs) if labs else None
        gv = Counter(r["genre"] for r in d if r["genre"])
        row["genre"] = sorted(gv, key=lambda g: (-gv[g], -len(g.split("|")), g))[0] if gv else None
        prov.append((cid, "genre", "vote" if gv and max(gv.values()) > 1 else "single" if gv else "none"))
        rows.append(row)
    Fz = pd.DataFrame(rows)[["_id"] + COLS]
    Fz["duration"] = Fz["duration"].astype("Int64")
    Fz["tracks"] = Fz["tracks"].map(lambda t: json.dumps(t, ensure_ascii=False) if t else None)
    Fz.to_pickle(f"{W}/state/fused.pkl")
    pd.DataFrame(prov, columns=["cluster_id", "attribute", "rule"]).to_csv(f"{W}/state/fusion_provenance.csv", index=False)
    # ---- export submission ----
    S = f"{ROOT}/submission"; os.makedirs(S, exist_ok=True)
    Fz.to_csv(f"{S}/fused.csv", index=False)
    M[["record_id", "source", "cluster_id"]].sort_values(["cluster_id", "record_id"]).to_csv(f"{S}/membership.csv", index=False)
    Cr = pd.read_pickle(f"{W}/state/correspondences.pkl")
    Cr[["id1", "id2", "score"]].to_csv(f"{S}/correspondences.csv", index=False)
    # ---- diagnostics (re-read saved files) ----
    fz = pd.read_csv(f"{S}/fused.csv", dtype=str); mb = pd.read_csv(f"{S}/membership.csv", dtype=str)
    cr = pd.read_csv(f"{S}/correspondences.csv", dtype=str); cand = pd.read_csv(f"{S}/blocking/candidates.csv", dtype=str)
    cs = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
    src_ids = set(N.id)
    import re
    diag = {"stage": "fusion", "ts": datetime.datetime.now().isoformat(), "inputs": "submission/*.csv",
            "fused_rows": len(fz), "fused_id_unique": bool(fz._id.is_unique),
            "membership_rows": len(mb), "records_total": len(src_ids), "membership_unknown_ids": int((~mb.record_id.isin(src_ids)).sum()),
            "membership_records_unique": bool(mb.record_id.is_unique),
            "clusters_without_fused_row": int((~mb.cluster_id.isin(fz._id)).sum()),
            "fused_rows_without_members": int((~fz._id.isin(mb.cluster_id)).sum()),
            "correspondences": len(cr), "correspondences_in_candidates": int(sum((a, b) in cs for a, b in zip(cr.id1, cr.id2))),
            "corr_same_cluster": int(sum(mb.set_index('record_id').cluster_id.get(a) == mb.set_index('record_id').cluster_id.get(b) for a, b in zip(cr.id1, cr.id2))),
            "density": {c: round(float(fz[c].notna().mean()), 4) for c in COLS},
            "date_pattern_valid": round(float(fz["release-date"].dropna().str.fullmatch(r"\d{4}-\d{2}-\d{2}").mean()), 4),
            "duration_int_valid": round(float(fz["duration"].dropna().str.fullmatch(r"\d+").mean()), 4),
            "genre_values": fz.genre.value_counts().head(8).to_dict(),
            "fusion_rule_counts": {f"{a}:{r}": int(n) for (a, r), n in pd.DataFrame(prov, columns=["c", "a", "r"]).groupby(["a", "r"]).size().items()}}
    diag = json.loads(json.dumps(diag, default=int))
    print(json.dumps(diag, indent=1))
    with open(f"{W}/diagnostics.jsonl", "a") as f:
        f.write(json.dumps(diag) + "\n")

if __name__ == "__main__":
    main()
