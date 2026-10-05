"""Stage 6: fusion. One row per membership cluster; values only from the cluster's own member records.
Outputs submission/{fused,membership,correspondences}.csv and state/fusion_provenance.csv
"""
import os, re, json, math, collections
import pandas as pd
from rapidfuzz import process, fuzz
from common import wsim, norm_title

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["dbpedia", "metacritic", "sales"]
L = {s: pd.read_csv(f"{W}/state/norm_{s}.csv", dtype=str, keep_default_na=False) for s in SRCS}
mem = pd.read_csv(f"{W}/state/membership.csv", dtype=str, keep_default_na=False)
recs = pd.concat(L.values(), ignore_index=True).set_index("id")
units = pd.read_csv(f"{W}/state/dbp_units.csv", dtype=str, keep_default_na=False).set_index("id")


# ------------------------------------------------------------------ global vocabulary repair
class Repair:
    """Map a rare noisy spelling to a frequent spelling of the same value (keyboard/OCR-aware similarity,
    order-insensitive). Frequencies are counted over all sources."""

    def __init__(self, values, min_freq=4, thr=0.84):
        self.freq = collections.Counter(v for v in values if v)
        self.frequent = [v for v, c in self.freq.items() if c >= min_freq]
        self.low = [v.lower() for v in self.frequent]
        self.min_freq, self.thr, self.cache = min_freq, thr, {}

    def __call__(self, v):
        if not v or self.freq[v] >= self.min_freq:
            return v
        if v in self.cache:
            return self.cache[v]
        lv = v.lower()
        best, bs = v, 0.0
        for m, sc, idx in process.extract(lv, self.low, scorer=fuzz.ratio, limit=8, score_cutoff=55):
            cand = self.frequent[idx]
            s = max(wsim(lv, m), wsim("".join(sorted(lv.split())), "".join(sorted(m.split()))))
            if s >= self.thr and (s > bs or (s == bs and self.freq[cand] > self.freq[best])):
                best, bs = cand, s
        self.cache[v] = best
        return best


def split_genres(s, src):
    if not s:
        return []
    parts = s.split(",") if src == "metacritic" else [s]
    return [p.strip() for p in parts if p.strip()]


dev_rep = Repair(pd.concat([L[s].developer for s in SRCS]))
pub_rep = Repair(L["sales"].publisher)
ser_rep = Repair(L["dbpedia"].series)
gen_vals = [g for s in SRCS for x in L[s].genres_raw for g in split_genres(x, s)]
gen_rep = Repair(gen_vals, min_freq=5, thr=0.8)

# genre -> taxonomy top-level names (non-exhaustive taxonomy; keep source genre as well)
tax = pd.read_csv(f"{ROOT}/task/input/schemamatching/Video_Game_Genres_Taxonomy.csv", dtype=str)
TAX_TOP = sorted(set(tax["Genre Name"]))
GEN_MAP = [  # (regex on lower-case source genre, taxonomy genre name)
    (r"shoot|fps|tps|shmup|gun", "Action"), (r"fight|beat|hack and slash|brawler", "Action"),
    (r"platform", "Action"), (r"stealth", "Action"), (r"^action$|action game|^act", "Action"),
    (r"action.adventure|action adventure|open.world|sandbox", "Action-Adventure"),
    (r"adventure|point.and.click|visual novel|interactive (fiction|movie)", "Adventure"),
    (r"role.playing|rpg|roguelike|dungeon crawl|mmorpg", "Role-Playing Game (RPG)"),
    (r"simulat|sim$|^sim|life sim|virtual life|management|tycoon|city.build|flight", "Simulation"),
    (r"strateg|tactic|rts|tbs|tower defense|4x|wargame|moba", "Strategy"),
    (r"sport|soccer|football|basketball|baseball|hockey|golf|tennis|wrestling|boxing|skate|snowboard|cricket",
     "Sports"),
    (r"racing|driving|kart|automobile|rally", "Racing"), (r"puzzle|match.3|tetris|trivia", "Puzzle"),
    (r"rhythm|music|danc", "Rhythm / Music"), (r"horror", "Horror"),
    (r"card|board|casino|poker|pinball|mahjong|chess|parlor", "Card & Board"),
    (r"educat|edutain", "Educational"), (r"party|minigame", "Party / Social"),
]


def tax_genres(g):
    lg = g.lower()
    out = []
    for pat, t in GEN_MAP:
        if re.search(pat, lg) and t not in out:
            out.append(t)
    if "action-adventure" in lg.replace(" ", "-") and "Action" in out and "Action-Adventure" in out:
        out.remove("Action")
    return out


# token frequencies for name quality
tokfreq = collections.Counter()
for s in SRCS:
    for t in set(L[s].ntitle):
        tokfreq.update(set(t.split()))


def name_quality(n):
    toks = norm_title(n).split()
    if not toks:
        return -99
    q = sum(1.0 if tokfreq[t] >= 3 else -1.0 for t in toks)
    if n.strip().endswith(":") or n.strip().endswith("-"):
        q -= 1.5  # truncated
    return q


def vote(items, key=lambda x: x, prio=("metacritic", "sales", "dbpedia")):
    """items: list of (value, source, weight). Majority by normalized key over per-source weights;
    tie-break by source priority, then value string."""
    if not items:
        return None, {}
    agg = collections.defaultdict(float)
    best_src = {}
    for v, s, w in items:
        k = key(v)
        agg[k] += w
        p = prio.index(s)
        if k not in best_src or p < best_src[k][0]:
            best_src[k] = (p, v)
    kbest = sorted(agg, key=lambda k: (-agg[k], best_src[k][0], str(k)))[0]
    return best_src[kbest][1], dict(agg)


# precompute per-record repaired values once (unique-value caches)
recs = recs.copy()
def _map_unique(series, f):
    u = {v: f(v) for v in series.unique()}
    return series.map(u)
recs["dev_r"] = _map_unique(recs["developer"], dev_rep)
recs["pub_r"] = _map_unique(recs["publisher"], pub_rep)
recs["ser_r"] = _map_unique(recs["series"], ser_rep)
_gcache = {}
def _genres(raw, src):
    k = (raw, src)
    if k not in _gcache:
        _gcache[k] = [gen_rep(x) for x in split_genres(raw, src)]
    return _gcache[k]
recs["gen_r"] = [_genres(r, s) for r, s in zip(recs.genres_raw, recs.source)]
REC = {rid: d for rid, d in zip(recs.index, recs.reset_index().to_dict("records"))}
_tax_cache = {}
def tax_cached(g):
    if g not in _tax_cache:
        _tax_cache[g] = tax_genres(g)
    return _tax_cache[g]
_nq = {}
def nq(n):
    if n not in _nq:
        _nq[n] = name_quality(n)
    return _nq[n]

rows, prov = [], []
clusters = collections.defaultdict(list)
for rid, cid in zip(mem.record_id, mem.cluster_id):
    clusters[cid].append(REC[rid])
PR = {"metacritic": 0, "sales": 1, "dbpedia": 2}
for cid in sorted(clusters):
    R = clusters[cid]
    D = [r for r in R if r["source"] == "dbpedia"]
    MS = [r for r in R if r["source"] != "dbpedia"]
    nd = len(D)
    wd = 1.0 / nd if nd else 0  # a dbpedia unit (many exploded rows) counts as one source vote
    def w(r):
        return wd if r["source"] == "dbpedia" else 1.0
    out = {"_id": cid}
    # ---- name: one candidate per metacritic/sales record + modal dbpedia spelling; best quality wins
    cands = [(r["name"], r["source"]) for r in MS if r["name"]]
    dn = collections.Counter(r["name"] for r in D if r["name"])
    if dn:
        cands.append((sorted(dn.items(), key=lambda kv: (-kv[1], kv[0]))[0][0], "dbpedia"))
    if cands:
        cands.sort(key=lambda x: (-nq(x[0]), PR[x[1]], x[0]))
        out["name"] = cands[0][0]; prov.append((cid, "name", cands[0][1]))
    # ---- platform
    out["platform"], _ = vote([(r["platform_label"], r["source"], w(r)) for r in R if r["platform_label"]])
    # ---- year: metacritic/sales version years; dbpedia only as fallback
    items = [(int(float(r["year"])), r["source"], 1.0) for r in MS if r["year"]]
    if items:
        v, _ = vote(items); yprov = "meta/sales vote"
    else:
        dis = sorted(int(r["dis_year"]) for r in D if r["dis_year"])
        ys = [int(float(r["year"])) for r in D if r["year"]]
        v = dis[0] if dis else (min(ys) if ys else None)
        yprov = "dbpedia disambiguation year" if dis else "dbpedia earliest year"
    out["releaseYear"] = f"{v:04d}-01-01" if v else None
    prov.append((cid, "releaseYear", yprov))
    # ---- developer (metacritic/sales preferred; dbpedia studio field is cross-product noisy)
    items = [(r["dev_r"], r["source"], 1.0) for r in MS if r["dev_r"]]
    if not items:
        items = [(r["dev_r"], r["source"], wd) for r in D if r["dev_r"]]
    out["developer"], _ = vote(items, key=norm_title)
    # ---- publisher
    out["publisher"], _ = vote([(r["pub_r"], r["source"], 1.0) for r in MS if r["pub_r"]], key=norm_title)
    # ---- genres: union of repaired source genres + taxonomy top-level mapping
    gcount = collections.Counter()
    for r in R:
        for gx in r["gen_r"]:
            gcount[gx] += w(r)
    taxs = collections.Counter()
    for gx, c in gcount.items():
        for t in tax_cached(gx):
            taxs[t] += c
    ordered = [t for t, _ in sorted(taxs.items(), key=lambda kv: (-kv[1], kv[0]))] + \
              [x for x, _ in sorted(gcount.items(), key=lambda kv: (-kv[1], kv[0]))]
    gl, seen = [], set()
    for x in ordered:
        if x.lower() not in seen:
            seen.add(x.lower()); gl.append(x)
    out["genres"] = json.dumps(gl[:10]) if gl else None
    # ---- scores / ESRB (metacritic + sales)
    for att, col in (("criticScore", "critic"), ("userScore", "user")):
        v, _ = vote([(float(r[col]), r["source"], 1.0) for r in MS if r[col]])
        out[att] = (int(round(v)) if att == "criticScore" else round(v, 1)) if v is not None else None
    out["ESRB"], _ = vote([(r["esrb"], r["source"], 1.0) for r in MS if r["esrb"]])
    # ---- series (dbpedia franchise)
    out["series"], _ = vote([(r["ser_r"], r["source"], 1.0) for r in D if r["ser_r"]], key=norm_title)
    out["id"] = cid
    rows.append(out)

cols = ["_id", "id", "name", "releaseYear", "developer", "genres", "publisher", "platform", "criticScore",
        "userScore", "ESRB", "series"]
fused = pd.DataFrame(rows)[cols]
fused["criticScore"] = fused["criticScore"].astype("Int64")
os.makedirs(f"{ROOT}/submission", exist_ok=True)
fused.to_csv(f"{ROOT}/submission/fused.csv", index=False)
mem[["record_id", "source", "cluster_id"]].to_csv(f"{ROOT}/submission/membership.csv", index=False)
pd.read_csv(f"{W}/state/correspondences.csv").to_csv(f"{ROOT}/submission/correspondences.csv", index=False)
pd.DataFrame(prov, columns=["cluster_id", "attribute", "rule"]).to_csv(f"{W}/state/fusion_provenance.csv", index=False)
print("fused rows", len(fused))
print(fused.notna().mean().round(3).to_dict())
