"""Stage 6: fusion + export of all submission files.
Per cluster, each source contributes ONE representative value per attribute (mode over that source's member rows,
so dbpedia's exploded rows don't outvote other sources). Values are compared by normalized key; the winning key is
the one supported by most sources, tie-break by source priority metacritic > sales > dbpedia (metacritic is the
cleanest-formatted source: ISO dates, list-valued fields). Genres: union (dedup by key) in priority order, capped at 10."""
import os, re, json, pandas as pd, numpy as np
from collections import Counter
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W); SUB = f"{ROOT}/submission"
R = pd.read_pickle(f"{W}/state/norm.pkl")
M = pd.read_pickle(f"{W}/state/membership.pkl")
R = R.merge(M[["rid","cluster_id"]], on="rid", how="left")
assert R.cluster_id.notna().all()
PRI = {"metacritic": 0, "sales": 1, "dbpedia": 2}
def k(v): return re.sub(r'[^a-z0-9]', '', str(v).lower())
PAREN = re.compile(r'\s*\([^()]*\)\s*$')
def mode(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals: return None
    c = Counter(vals); return sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
def vote(cands, keyf=k):
    """cands: list of (source, value). One vote per source; tie -> source priority."""
    cands = [(s, v) for s, v in cands if v is not None]
    if not cands: return None, None
    cnt = Counter(keyf(v) for _, v in cands)
    best = sorted(cands, key=lambda sv: (-cnt[keyf(sv[1])], PRI[sv[0]]))[0]
    return best[1], best[0]
def date_ok(d): return d is not None and "1960-01-01" <= d <= "2024-12-31"
out, prov = [], []
for cid, g in R.groupby("cluster_id", sort=True):
    per = {s: x for s, x in g.groupby("source")}
    multi = len(per) > 1
    rep = {}
    for s, x in per.items():
        devs = [", ".join(l) if s != "dbpedia" else (l[0] if l else None) for l in x.dev_list]
        rep[s] = dict(name=mode(x.name_c.tolist()), platform=mode(x.platform_c.tolist()),
                      date=mode([d for d in x.date_c if date_ok(d)]),
                      dev=mode([d for d in devs if d]), pub=mode(x.pub_c.tolist()),
                      critic=mode(x.critic_c.tolist()), user=mode(x.user_c.tolist()),
                      esrb=mode(x.esrb_c.tolist()), series=mode(x.series_c.tolist()),
                      genres=[gg for l in x.genre_list for gg in l])
    row = {"_id": cid}; pv = {"_id": cid, "sources": ",".join(sorted(per))}
    def pick(attr, keyf=k):
        v, s = vote([(s, r[attr]) for s, r in rep.items()], keyf); pv[attr] = s; return v
    row["id"] = cid
    row["name"] = pick("name")
    row["platform"] = pick("platform")
    row["releaseYear"] = pick("date", keyf=lambda d: d[:4])
    # developer: compare with dbpedia disambiguation stripped; dbpedia spelling only if no other source has one
    dv, ds = vote([(s, r["dev"]) for s, r in rep.items()], keyf=lambda v: k(PAREN.sub('', v)))
    row["developer"] = dv; pv["dev"] = ds
    gl, seen = [], set()
    for s in sorted(rep, key=PRI.get):
        for gg in rep[s]["genres"]:
            if k(gg) and k(gg) not in seen: seen.add(k(gg)); gl.append(gg)
    row["genres"] = json.dumps(gl[:10]) if gl else None
    row["publisher"] = pick("pub")
    cv = pick("critic", keyf=lambda v: v); row["criticScore"] = int(round(cv)) if cv is not None else None
    uv = pick("user", keyf=lambda v: v); row["userScore"] = round(float(uv), 1) if uv is not None else None
    row["ESRB"] = pick("esrb", keyf=lambda v: v)
    row["series"] = pick("series")
    out.append(row); prov.append(pv)
F = pd.DataFrame(out)
cols = ["_id","id","name","releaseYear","developer","genres","publisher","platform","criticScore","userScore","ESRB","series"]
F = F[cols]
F["criticScore"] = F.criticScore.astype("Int64")
os.makedirs(f"{SUB}/blocking", exist_ok=True)
F.to_csv(f"{SUB}/fused.csv", index=False)
pd.DataFrame(prov).to_csv(f"{W}/state/fusion_provenance.csv", index=False)
M[["rid","source","cluster_id"]].rename(columns={"rid":"record_id"}).sort_values(["cluster_id","record_id"]).to_csv(f"{SUB}/membership.csv", index=False)
P = pd.read_pickle(f"{W}/state/correspondences.pkl"); P.to_csv(f"{SUB}/correspondences.csv", index=False)
print("fused rows", len(F)); print(F.notna().mean().round(3).to_dict())
