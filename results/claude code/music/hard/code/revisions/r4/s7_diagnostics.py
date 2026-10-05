"""Label-free diagnostic panel computed from the saved submission artifacts."""
import os, sys, json, re, datetime
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = f"{BASE}/submission"
rev = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
src = {s: pd.read_csv(f"{BASE}/task/input/data/{s}.csv", dtype=str, keep_default_na=False) for s in ["discogs", "lastfm", "musicbrainz"]}
ids = {i: s for s, d in src.items() for i in d.id}
mem = pd.read_csv(f"{SUB}/membership.csv", dtype=str)
fus = pd.read_csv(f"{SUB}/fused.csv", dtype=str, keep_default_na=False)
cor = pd.read_csv(f"{SUB}/correspondences.csv", dtype=str)
cand = pd.read_csv(f"{SUB}/blocking/candidates.csv", dtype=str)
tax = pd.read_csv(f"{BASE}/task/input/schemamatching/Music_Genres_Taxonomy.csv")
cs = set(map(tuple, map(sorted, zip(cand.id1, cand.id2))))
sizes = mem.groupby("cluster_id").size()
nsrc = mem.groupby("cluster_id").source.nunique()
d = dict(
    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(), revision=rev, inputs="submission/*.csv",
    records_total=len(ids), membership_rows=len(mem), membership_unique=int(mem.record_id.nunique()),
    unresolved_ids=int((~mem.record_id.isin(ids)).sum()) + int((~cor.id1.isin(ids)).sum() + (~cor.id2.isin(ids)).sum()),
    source_label_mismatch=int((mem.record_id.map(ids) != mem.source).sum()),
    missing_records=len(set(ids) - set(mem.record_id)),
    fused_rows=len(fus), fused_ids_eq_clusters=set(fus._id) == set(mem.cluster_id),
    corr=len(cor), corr_in_candidates=float(sum(tuple(sorted(p)) in cs for p in zip(cor.id1, cor.id2)) / max(len(cor), 1)),
    corr_same_source=int((cor.id1.map(ids) == cor.id2.map(ids)).sum()),
    corr_consistent_with_membership=float((cor.id1.map(dict(zip(mem.record_id, mem.cluster_id))) ==
                                           cor.id2.map(dict(zip(mem.record_id, mem.cluster_id)))).mean()),
    candidates=len(cand),
    cluster_size_dist={int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
    max_same_source_in_cluster=int(mem.groupby(["cluster_id", "source"]).size().max()),
    singleton_share=round(float((sizes == 1).mean()), 4),
    matched_share_by_source={s: round(float(mem[mem.source == s].cluster_id.map(sizes).gt(1).mean()), 4) for s in src},
    density={c: round(float((fus[c].str.strip() != "").mean()), 4) for c in fus.columns},
    date_valid=float(fus["release-date"][fus["release-date"] != ""].str.fullmatch(r"\d{4}-\d{2}-\d{2}").mean()),
    duration_int=float(fus.duration[fus.duration != ""].str.fullmatch(r"\d+").mean()),
    genre_in_taxonomy_any_level=float(fus.genre[fus.genre != ""].isin(set(tax["Genre Name"]) | set(tax["Subgenre Name"]) | set(tax["Sub-Subgenre Name"])).mean()),
    tracks_json=float(fus.tracks[fus.tracks != ""].map(lambda s: s.startswith("[")).mean()),
)
print(json.dumps(d, indent=1))
open(f"{BASE}/work/diagnostics.jsonl", "a").write(json.dumps(d) + "\n")
