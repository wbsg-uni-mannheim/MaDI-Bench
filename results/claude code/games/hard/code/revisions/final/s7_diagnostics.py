"""Label-free diagnostic panel over the saved submission artifacts (not correctness scores)."""
import os, sys, json, re, time, collections
import pandas as pd

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
S = f"{ROOT}/submission"
rev = sys.argv[1] if len(sys.argv) > 1 else "current"
src_ids = {s: set(pd.read_csv(f"{ROOT}/task/input/data/{s}.csv", dtype=str, usecols=["id"]).id)
           for s in ["dbpedia", "metacritic", "sales"]}
mem = pd.read_csv(f"{S}/membership.csv", dtype=str)
cor = pd.read_csv(f"{S}/correspondences.csv", dtype=str)
fus = pd.read_csv(f"{S}/fused.csv", dtype=str, keep_default_na=False)
cand = pd.read_csv(f"{S}/blocking/candidates.csv", dtype=str)
allids = set().union(*src_ids.values())
d = {"rev": rev, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": "submission/*.csv"}
d["records_in_membership"] = {s: int(mem[mem.source == s].record_id.isin(src_ids[s]).sum()) for s in src_ids}
d["source_records"] = {s: len(v) for s, v in src_ids.items()}
d["membership_dupe_ids"] = int(mem.record_id.duplicated().sum())
d["unresolved_ids"] = int((~mem.record_id.isin(allids)).sum() + (~cor.id1.isin(allids)).sum() + (~cor.id2.isin(allids)).sum())
cs = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
d["correspondences"] = len(cor)
d["corr_in_candidates"] = int(sum((a, b) in cs for a, b in zip(cor.id1, cor.id2)))
d["candidates"] = len(cand)
sz = mem.groupby("cluster_id").size()
d["clusters"] = int(len(sz)); d["max_cluster"] = int(sz.max())
d["singleton_share"] = round(float((sz == 1).mean()), 4)
ns = mem.groupby("cluster_id").source.agg(lambda x: "+".join(sorted(set(x))))
d["cluster_source_mix"] = ns.value_counts().to_dict()
multi = mem.groupby(["cluster_id", "source"]).size()
d["clusters_with_2plus_meta_or_sales"] = int(multi[(multi.index.get_level_values(1) != "dbpedia") & (multi > 1)].shape[0])
d["fused_ids_eq_clusters"] = set(fus._id) == set(mem.cluster_id)
d["fused_rows"] = len(fus)
d["density"] = {c: round(float((fus[c] != "").mean()), 4) for c in fus.columns if c != "_id"}
# schema validity
ok_year = fus.releaseYear.map(lambda v: v == "" or bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)) and "1960" <= v[:4] <= "2024")
ok_esrb = fus.ESRB.isin(["", "E", "E10+", "T", "M", "AO", "RP", "RP-LM17"])
ok_cs = fus.criticScore.map(lambda v: v == "" or (v.isdigit() and 0 <= int(v) <= 100))
ok_us = fus.userScore.map(lambda v: v == "" or 0 <= float(v) <= 10)
ok_gen = fus.genres.map(lambda v: v == "" or (1 <= len(json.loads(v)) <= 10))
d["schema_invalid"] = {"releaseYear": int((~ok_year).sum()), "ESRB": int((~ok_esrb).sum()),
                       "criticScore": int((~ok_cs).sum()), "userScore": int((~ok_us).sum()),
                       "genres": int((~ok_gen).sum())}
tax = set(pd.read_csv(f"{ROOT}/task/input/schemamatching/Gaming_Platforms_Taxonomy.csv", dtype=str)["Platform Name"])
d["platform_in_taxonomy_share"] = round(float(fus[fus.platform != ""].platform.isin(tax).mean()), 4)
# traceability: fused name tokens present in some member's raw name
raw = {}
for s in src_ids:
    t = pd.read_csv(f"{W}/state/translated_{s}.csv", dtype=str, keep_default_na=False)
    raw.update(dict(zip(t.id, t.name)))
tok = lambda x: set(re.findall(r"[a-z0-9]+", x.lower()))
fn = dict(zip(fus._id, fus.name))
bad = 0; tot = 0
for c, g in mem.groupby("cluster_id"):
    n = fn.get(c, "")
    if not n:
        continue
    tot += 1
    if not any(tok(n) <= tok(raw.get(r, "")) for r in g.record_id):
        bad += 1
d["fused_name_not_traceable"] = bad
print(json.dumps(d, indent=1, default=str))
with open(f"{W}/diagnostics.jsonl", "a") as f:
    f.write(json.dumps(d, default=str) + "\n")
