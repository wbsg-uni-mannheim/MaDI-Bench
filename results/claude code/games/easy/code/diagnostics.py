"""Label-free structural diagnostics on the SAVED submission files. Appends to diagnostics.jsonl."""
import os, json, datetime, pandas as pd
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W); S = f"{ROOT}/submission"
M = pd.read_csv(f"{S}/membership.csv", dtype=str); F = pd.read_csv(f"{S}/fused.csv", dtype=str)
P = pd.read_csv(f"{S}/correspondences.csv", dtype=str); C = pd.read_csv(f"{S}/blocking/candidates.csv", dtype=str)
src_ids = {s: set(pd.read_csv(f"{ROOT}/task/input/data/{s}.csv", dtype=str).id) for s in ["dbpedia","metacritic","sales"]}
d = {}
d["records_total"] = sum(len(v) for v in src_ids.values())
d["membership_rows"] = len(M); d["membership_unique"] = M.record_id.nunique()
d["unresolved_ids"] = int(sum(~M.apply(lambda r: r.record_id in src_ids.get(r.source, ()), axis=1)))
d["fused_rows"] = len(F); d["fused_ids_eq_clusters"] = set(F._id) == set(M.cluster_id)
cand = set(map(tuple, C[["id1","id2"]].apply(sorted, axis=1).tolist()))
d["corr_pairs"] = len(P); d["corr_in_candidates"] = float(pd.Series([tuple(sorted(x)) in cand for x in P[["id1","id2"]].values]).mean()) if len(P) else 1.0
d["candidate_pairs"] = len(C)
sz = M.cluster_id.value_counts(); ns = M.groupby("cluster_id").source.nunique()
d["cluster_size_dist"] = sz.value_counts().sort_index().head(12).to_dict()
d["clusters_gt9"] = int((sz > 9).sum()); d["share_records_in_gt9"] = round(float(sz[sz > 9].sum() / len(M)), 4)
d["sources_per_cluster"] = ns.value_counts().sort_index().to_dict()
d["singleton_share"] = round(float((sz == 1).mean()), 4)
d["density"] = F.notna().mean().round(3).to_dict()
esrb_ok = {"E","E10+","T","M","AO","RP","RP-LM17"}
d["esrb_invalid"] = int((~F.ESRB.dropna().isin(esrb_ok)).sum())
yr = F.releaseYear.dropna(); d["date_invalid"] = int((~yr.str.match(r"^\d{4}-\d{2}-\d{2}$") | (yr > "2024-12-31") | (yr < "1960-01-01")).sum())
cs = pd.to_numeric(F.criticScore, errors="coerce"); us = pd.to_numeric(F.userScore, errors="coerce")
d["score_out_of_range"] = int(((cs < 0) | (cs > 100)).sum() + ((us < 0) | (us > 10)).sum())
d["genres_over10"] = int(F.genres.dropna().map(lambda g: len(json.loads(g)) > 10).sum())
d["ts"] = datetime.datetime.now().isoformat(timespec="seconds"); d["inputs"] = "submission/*.csv"
d["revision"] = open(f"{W}/REVISION").read().strip() if os.path.exists(f"{W}/REVISION") else "working"
with open(f"{W}/diagnostics.jsonl", "a") as f: f.write(json.dumps(d, default=str) + "\n")
print(json.dumps(d, indent=1, default=str))
