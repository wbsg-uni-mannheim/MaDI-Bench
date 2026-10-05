"""Label-free diagnostic panel computed from the saved submission artifacts."""
import pandas as pd, json, re, sys, datetime, random
rev = sys.argv[1] if len(sys.argv) > 1 else "current"
S = "submission/"
FU = pd.read_csv(S + "fused.csv", dtype=str, keep_default_na=False)
ME = pd.read_csv(S + "membership.csv", dtype=str)
CO = pd.read_csv(S + "correspondences.csv", dtype={"id1": str, "id2": str})
CA = pd.read_csv(S + "blocking/candidates.csv", dtype=str)
R = pd.read_pickle("work/state/records.pkl").set_index("id")
src_ids = {}
for s, col in [("discogs", "rec_uid"), ("lastfm", "item_code"), ("musicbrainz", "Attribute_1")]:
    src_ids[s] = set(pd.read_csv(f"task/input/data/{s}.csv", dtype=str)[col])
allids = set().union(*src_ids.values())
tax = set(pd.read_csv("task/input/schemamatching/Music_Genres_Taxonomy.csv")["Genre Name"])
d = {}
d["membership_rows"] = len(ME); d["source_records"] = len(allids)
d["unresolved_membership_ids"] = int((~ME.record_id.isin(allids)).sum())
d["records_missing_from_membership"] = len(allids - set(ME.record_id))
d["dup_membership"] = int(ME.record_id.duplicated().sum())
d["membership_source_ok"] = bool(all(r in src_ids[s] for r, s in zip(ME.record_id, ME.source)))
d["cluster_ids_not_in_fused"] = len(set(ME.cluster_id) - set(FU._id)); d["fused_ids_not_in_membership"] = len(set(FU._id) - set(ME.cluster_id))
d["fused_rows"] = len(FU); d["fused_id_unique"] = bool(FU._id.is_unique)
cl = dict(zip(ME.record_id, ME.cluster_id)); sr = dict(zip(ME.record_id, ME.source))
d["corr"] = len(CO); d["corr_same_cluster"] = float((CO.id1.map(cl) == CO.id2.map(cl)).mean())
d["corr_cross_source"] = float((CO.id1.map(sr) != CO.id2.map(sr)).mean())
cs = set(zip(CA.id1, CA.id2)) | set(zip(CA.id2, CA.id1))
d["corr_in_candidates"] = float(pd.Series([(a, b) in cs for a, b in zip(CO.id1, CO.id2)]).mean())
d["candidates"] = len(CA)
sz = ME.groupby("cluster_id").size(); ns = ME.groupby("cluster_id").source.nunique()
d["cluster_size_dist"] = sz.value_counts().sort_index().to_dict(); d["max_within_source_multiplicity"] = int(ME.groupby(["cluster_id", "source"]).size().max())
d["singleton_share"] = round(float((sz == 1).mean()), 4)
d["linked_share_by_source"] = {s: round(float((ME[ME.source == s].cluster_id.map(sz) > 1).mean()), 4) for s in src_ids}
d["composition"] = ME.groupby("cluster_id").source.apply(lambda x: "+".join(sorted(x))).value_counts().to_dict()
# validity + density
blank = lambda c: FU[c].str.strip() == ""
dens = {c: round(float((~blank(c)).mean()), 4) for c in FU.columns}
d["density"] = dens
multi = FU[FU._id.map(sz) > 1]
d["density_multi_source_rows"] = {c: round(float((multi[c].str.strip() != "").mean()), 4) for c in FU.columns}
dt = FU["release-date"][~blank("release-date")]
d["date_invalid"] = int((~dt.str.fullmatch(r"\d{4}-\d{2}-\d{2}") | (dt < "1900-01-01") | (dt > "2016-12-31")).sum())
g = FU.genre[~blank("genre")]; d["genre_not_in_taxonomy"] = int((~g.isin(tax)).sum())
du = FU.duration[~blank("duration")].astype(float); d["duration_invalid"] = int(((du < 1) | (du > 86400) | (du != du.round())).sum())
tr = FU.tracks[~blank("tracks")].map(json.loads); d["tracks_invalid"] = int(tr.map(lambda x: not (1 <= len(x) <= 500) or any(not t or len(t) > 500 for t in x)).sum())
d["name_too_long"] = int((FU.name.str.len() > 512).sum()); d["label_too_long"] = int((FU.label.str.len() > 300).sum())
d["country_too_short"] = int(((FU["release-country"].str.len() < 2) & ~blank("release-country")).sum())
# traceability: every fused value must equal (or be the documented transform of) some member's value
mem = ME.groupby("cluster_id").record_id.apply(list).to_dict()
bad = {k: 0 for k in ["name", "artist", "genre", "label", "duration", "release-country", "tracks", "release-date"]}
for _, r in FU.iterrows():
    ms = mem[r._id]
    if r["name"] and r["name"] not in {R.at[i, "name"] for i in ms}: bad["name"] += 1
    if r.artist and r.artist not in {R.at[i, "artist"] for i in ms}: bad["artist"] += 1
    if r.genre and r.genre not in {R.at[i, "genre"] for i in ms}: bad["genre"] += 1
    if r.label and r.label not in {"|".join(R.at[i, "labels"]) for i in ms}: bad["label"] += 1
    if r.duration and int(r.duration) not in {R.at[i, "duration"] for i in ms}: bad["duration"] += 1
    if r["release-country"] and r["release-country"] not in {R.at[i, "country"] for i in ms}: bad["release-country"] += 1
    if r.tracks and json.loads(r.tracks) not in [R.at[i, "tracks"] for i in ms]: bad["tracks"] += 1
    if r["release-date"] and not any(R.at[i, "date"] and r["release-date"].startswith(R.at[i, "date"]) for i in ms): bad["release-date"] += 1
d["untraceable_values"] = bad
# disagreement rates within multi-source clusters
P = pd.read_csv("work/state/fusion_provenance.csv")
dis = {}
for c, col in [("artist", "artist"), ("name", "name"), ("country", "country")]:
    n = k = 0
    for ms in mem.values():
        if len(ms) < 2: continue
        vals = {re.sub(r"[^a-z0-9]", "", str(R.at[i, col]).lower()) for i in ms if R.at[i, col]}
        if len(vals) >= 1: n += 1; k += len(vals) > 1
    dis[c] = round(k / max(n, 1), 4)
d["multi_cluster_disagreement_rate"] = dis
print(json.dumps(d, indent=1, default=str))
d.update(stage="panel", revision=rev, ts=datetime.datetime.now().isoformat(), inputs="submission/*.csv")
open("work/diagnostics.jsonl", "a").write(json.dumps(d, default=str) + "\n")
