"""Stage 7: fusion from final clusters + write submission files."""
import pandas as pd, numpy as np, json, re, os
R = pd.read_pickle("work/state/records.pkl").set_index("id")
M = pd.read_pickle("work/state/clusters.pkl")
C = pd.read_pickle("work/state/corr.pkl")
PRI = {"discogs": 0, "musicbrainz": 1, "lastfm": 2}          # default priority (discogs is the least noisy source)
DUR_PRI = {"discogs": 0, "lastfm": 1, "musicbrainz": 2}      # duration: schema example (903) follows lastfm over mb
def key(s): return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())
def by_pri(mem, pri=PRI): return sorted(mem, key=lambda i: (pri[R.source[i]], i))
def first(mem, col, pri=PRI):
    for i in by_pri(mem, pri):
        v = R.at[i, col]
        if isinstance(v, list) and v: return v, i
        if isinstance(v, str) and v.strip(): return v, i
    return None, None
def vote_text(mem, col):
    """vote over normalized key; ties -> source priority; emit spelling of best-priority supporter"""
    groups = {}
    for i in by_pri(mem):
        v = R.at[i, col]
        if v and v.strip(): groups.setdefault(key(v) or v, []).append(i)
    if not groups: return None, None
    best = sorted(groups.values(), key=lambda g: (-len({R.source[i] for i in g}), PRI[R.source[g[0]]]))[0]
    return R.at[best[0], col], best[0]
def fuse_date(mem):
    d = [i for i in by_pri(mem) if R.at[i, "date"]]
    for i in d:                         # priority order; complete to calendar date by source convention
        v, p = R.at[i, "date"], R.at[i, "date_prec"]
        if p == "day": out = v
        elif p == "month": out = v + "-01"
        else: out = v + "-01-01"
        if "1900-01-01" <= out <= "2016-12-31": return out, i
    return None, None
def fuse_duration(mem):
    vals = [(i, R.at[i, "duration"]) for i in mem if pd.notna(R.at[i, "duration"]) and 0 < R.at[i, "duration"] <= 86400]
    if not vals: return None, None
    groups = []
    for i, v in vals:
        g = [j for j, w in vals if abs(w - v) <= 5]
        groups.append((len({R.source[j] for j in g}), -DUR_PRI[R.source[i]], i, v))
    n, _, i, v = max(groups)
    return int(v), i
rows, prov = [], []
for root, mem in M.groupby("root").record_id:
    mem = list(mem)
    cid = min(mem, key=lambda i: (PRI[R.source[i]], i))   # cluster id = a member's native id (stable)
    out = {"_id": cid, "id": cid}
    src = {}
    out["name"], src["name"] = first(mem, "name")
    out["artist"], src["artist"] = vote_text(mem, "artist")
    out["release-date"], src["release-date"] = fuse_date(mem)
    out["release-country"], src["release-country"] = first(mem, "country")
    lab, src["label"] = first(mem, "labels"); out["label"] = "|".join(lab) if lab else None
    out["genre"], src["genre"] = first(mem, "genre")
    tr, src["tracks"] = first(mem, "tracks"); out["tracks"] = json.dumps(tr[:500], ensure_ascii=False) if tr else None
    out["duration"], src["duration"] = fuse_duration(mem)
    rows.append(out); prov.append(dict(_id=cid, members="|".join(sorted(mem)), **{f"src_{k}": v for k, v in src.items()}))
FU = pd.DataFrame(rows)
cmap = {}
for root, mem in M.groupby("root").record_id:
    cid = min(mem, key=lambda i: (PRI[R.source[i]], i))
    for i in mem: cmap[i] = cid
M["cluster_id"] = M.record_id.map(cmap)
os.makedirs("submission", exist_ok=True)
cols = ["_id", "id", "name", "artist", "release-date", "release-country", "label", "genre", "tracks", "duration"]
FU["duration"] = FU["duration"].astype("Int64")
FU[cols].to_csv("submission/fused.csv", index=False)
M[["record_id", "source", "cluster_id"]].to_csv("submission/membership.csv", index=False)
C.to_csv("submission/correspondences.csv", index=False)
pd.DataFrame(prov).to_csv("work/state/fusion_provenance.csv", index=False)
print(len(FU), FU.notna().mean().round(3).to_dict())
