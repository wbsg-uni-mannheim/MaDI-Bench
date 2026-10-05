"""Stage 4: entity matching (rule-based, no training).
Nodes: metacritic records, sales records, dbpedia *units* = (dbpedia article, platform).
A dbpedia article = rows sharing a normalized title (+ disambiguation year), with rare noisy title variants
folded into the dominant spelling.
Outputs: state/dbp_units.csv (dbpedia row -> unit), state/pair_scores.csv (all scored node pairs + decision).
"""
import os, re, json, time, collections
import pandas as pd
from rapidfuzz import fuzz
from common import wsim, numbers_compatible

W = os.path.dirname(os.path.abspath(__file__))
GENERIC = {"the", "soccer", "video", "game", "videogame", "edition", "a"}
P = dict(  # decision settings (documented in report)
    variant_ws=0.90, variant_ratio=3,
    acc=[(0.97, -0.5), (0.88, 1.0), (0.80, 2.0), (0.70, 2.5)],
    subst_ev=3.0,                      # word-substitution titles need scores+year agreement
    dup=[(0.999, 1.0), (0.97, 1.5), (0.80, 2.5)],    # same-source duplicate (metacritic/sales) acceptance

)

L = {s: pd.read_csv(f"{W}/state/norm_{s}.csv", dtype=str, keep_default_na=False)
     for s in ["dbpedia", "metacritic", "sales"]}
tp = pd.read_csv(f"{W}/state/title_pairs.csv", keep_default_na=False)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


TOKFREQ = collections.Counter()
for _s in L:
    for _t in set(L[_s].ntitle):
        TOKFREQ.update(set(_t.split()))


def word_subst(a, b):
    """True when each title has a real (frequent, len>=3) word the other lacks, i.e. one word was replaced
    by a different word (autobots/decepticons, nfl/nhl) rather than typo-noised, truncated or dropped."""
    if wsim(a.replace(" ", ""), b.replace(" ", "")) >= 0.9 and len(a.split()) != len(b.split()):
        return False  # word-boundary differences (roadblasters / road blaster)
    ta, tb = a.split(), b.split()
    used = set()
    ua = []
    for x in ta:
        hit = None
        for j, y in enumerate(tb):
            if j not in used and (x == y or wsim(x, y) >= 0.7 or
                                  (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x)))):
                hit = j; break
        if hit is None:
            ua.append(x)
        else:
            used.add(hit)
    ub = [y for j, y in enumerate(tb) if j not in used]
    real = lambda ws: any(len(w) >= 3 for w in ws)
    return real(ua) and real(ub)


_ts_cache = {}


def title_sim(a, b):
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    if key in _ts_cache:
        return _ts_cache[key]
    if not numbers_compatible(a, b):
        s = 0.0
    elif max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) < 60:
        s = 0.0
    else:
        s1 = wsim(a.replace(" ", ""), b.replace(" ", ""))
        s2 = wsim("".join(sorted(a.split())), "".join(sorted(b.split())))
        s = max(s1, s2)
        ga = sorted(w for w in a.split() if w not in GENERIC)
        gb = sorted(w for w in b.split() if w not in GENERIC)
        if ga and ga == gb:
            s = max(s, 0.97)
    _ts_cache[key] = s
    return s


# ------------------------------------------------------------------ dbpedia articles & units
d = L["dbpedia"].copy()
d["akey"] = d["ntitle"] + "|" + d["dis_year"]
cnt = d[d.ntitle != ""].groupby("akey").size()
akeys_by_title = collections.defaultdict(list)
for k in cnt.index:
    akeys_by_title[k.split("|")[0]].append(k)
parent = {k: k for k in cnt.index}


def find(k):
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


# fold rare noisy variants into dominant spelling (same disambiguation year)
merges = []
for t1, t2 in zip(tp.t1, tp.t2):
    if t1 not in akeys_by_title or t2 not in akeys_by_title:
        continue
    s = title_sim(t1, t2)
    if s < P["variant_ws"]:
        continue
    for k1 in akeys_by_title[t1]:
        for k2 in akeys_by_title[t2]:
            if k1.split("|")[1] != k2.split("|")[1]:
                continue
            big, small = (k1, k2) if cnt[k1] >= cnt[k2] else (k2, k1)
            if cnt[big] >= P["variant_ratio"] * cnt[small]:
                merges.append((small, big, s))
for small, big, s in sorted(merges, key=lambda x: -x[2]):
    rs, rb = find(small), find(big)
    if rs != rb:
        parent[rs] = rb
d["article"] = d["akey"].map(lambda k: find(k) if k in parent else "")
d["unit"] = ""
has = (d.article != "") & (d.pkey != "")
d.loc[has, "unit"] = "U:" + d.loc[has, "article"] + "|" + d.loc[has, "pkey"]
# rows with unknown platform: attach only if their article has exactly one platform unit
units_per_article = d[has].groupby("article")["unit"].agg(lambda x: sorted(set(x)))
one = units_per_article[units_per_article.map(len) == 1].map(lambda x: x[0])
unk = (d.article != "") & (d.pkey == "")
d["unit_how"] = ""
d.loc[has, "unit_how"] = "platform"
att = unk & d.article.isin(one.index)
d.loc[att, "unit"] = d.loc[att, "article"].map(one)
d.loc[att, "unit_how"] = "unique_platform_of_article"
# remaining unknown-platform rows: one '?' unit per article. Matchable only when the article has no known
# platform at all (then the unit may attach to a single clearly-supported entity in clustering phase 2).
rest = unk & (d.unit == "")
d.loc[rest, "unit"] = "U:" + d.loc[rest, "article"] + "|?"
no_known = ~d.article.isin(units_per_article.index)
d.loc[rest & no_known, "unit_how"] = "article_without_platform"
d.loc[rest & ~no_known, "unit"] = ""   # ambiguous platform: kept as singletons
d.loc[rest & ~no_known, "unit_how"] = "unknown_platform_rows_of_multi_platform_article"
d[["id", "article", "unit", "unit_how"]].to_csv(f"{W}/state/dbp_units.csv", index=False)
print("dbpedia articles", d.article.nunique(), "units", d[d.unit != ""].unit.nunique(),
      "variant merges", len(merges), "rows w/o unit", int((d.unit == "").sum()))

# ------------------------------------------------------------------ node table
nodes = {}
for u, g in d[(d.unit != "") & (d.unit_how != "unknown_platform_rows_of_multi_platform_article")].groupby("unit"):
    art = g.article.iloc[0]
    pk = u.rsplit("|", 1)[1]
    nodes[u] = dict(src="dbpedia", ntitle=art.split("|")[0], dis_year=fnum(art.split("|")[1]),
                    pkey="" if pk == "?" else pk, years={int(float(y)) for y in g.year if y},
                    devs={x for x in g.dev_key if x}, critic=None, user=None, n=len(g))
for s in ["metacritic", "sales"]:
    for r in L[s].itertuples(index=False):
        if not r.ntitle:
            continue
        nodes[r.id] = dict(src=s, ntitle=r.ntitle, dis_year=None, pkey=r.pkey,
                           years={int(float(r.year))} if r.year else set(), devs={r.dev_key} if r.dev_key else set(),
                           critic=fnum(r.critic), user=fnum(r.user), n=1)
by_title = collections.defaultdict(list)
for nid, nd in nodes.items():
    by_title[nd["ntitle"]].append(nid)
print("nodes", collections.Counter(n["src"] for n in nodes.values()))


def dev_sim(A, B):
    best = 0.0
    for a in A:
        for b in B:
            best = max(best, fuzz.token_set_ratio(a, b) / 100.0)
    return best


def evidence(a, b):
    """returns (evidence_sum, detail dict). a,b node dicts of different sources."""
    ev, det = 0.0, {}
    # year
    if a["src"] != "dbpedia" and b["src"] != "dbpedia":
        if a["years"] and b["years"]:
            dy = abs(next(iter(a["years"])) - next(iter(b["years"])))
            v = 1.0 if dy == 0 else 0.5 if dy == 1 else -0.5 if dy == 2 else -1.5
            ev += v; det["year"] = v
    else:
        dbn, oth = (a, b) if a["src"] == "dbpedia" else (b, a)
        if oth["years"]:
            y = next(iter(oth["years"]))
            v = 0.0
            if dbn["years"] and min(abs(y - z) for z in dbn["years"]) <= 1:
                v = 1.0
            if dbn["dis_year"] and y < dbn["dis_year"] - 1:
                v = -1.5
            if dbn["dis_year"] and abs(y - dbn["dis_year"]) <= 1:
                v = max(v, 1.0)
            ev += v; det["year"] = v
    # developer
    if a["devs"] and b["devs"]:
        ds = dev_sim(a["devs"], b["devs"])
        v = 1.0 if ds >= 0.85 else 0.0
        if v == 0 and a["src"] != "dbpedia" and b["src"] != "dbpedia" and ds < 0.5:
            v = -0.3
        ev += v; det["dev"] = v
    # scores (metacritic <-> sales only)
    if a["critic"] is not None and b["critic"] is not None:
        dc = abs(a["critic"] - b["critic"])
        v = 1.0 if dc <= 1 else 0.0 if dc <= 5 else -1.0
        ev += v; det["critic"] = v
    if a["user"] is not None and b["user"] is not None:
        du = abs(a["user"] - b["user"])
        v = 0.7 if du <= 0.1 else 0.0 if du < 1.0 else -0.5
        ev += v; det["user"] = v
    return ev, det


def decide(ts, ev, unknown_platform, subst=False):
    if subst and ts < 1.0:
        return ev >= P["subst_ev"] and ts >= 0.8
    if unknown_platform:
        return ts >= 0.88 and ev >= 2.0
    return any(ts >= t and ev >= e for t, e in P["acc"])


# ------------------------------------------------------------------ candidate node pairs
tpairs = [(t, t) for t in by_title] + list(zip(tp.t1, tp.t2))
rows = []
t0 = time.time()
for t1, t2 in tpairs:
    n1, n2 = by_title.get(t1), by_title.get(t2)
    if not n1 or not n2:
        continue
    ts = title_sim(t1, t2)
    if ts < 0.70:
        continue
    for x in n1:
        for y in n2:
            if t1 == t2 and x >= y:
                continue
            a, b = nodes[x], nodes[y]
            pa, pb = a["pkey"], b["pkey"]
            if pa and pb and pa != pb:
                continue
            if not pa and not pb:
                continue
            unknown = not (pa and pb)
            sub = word_subst(t1, t2) if t1 != t2 else False
            if a["src"] == b["src"]:
                if a["src"] == "dbpedia" or unknown:
                    continue
                ev, det = evidence(a, b)
                neg = any(v <= -0.5 for v in det.values())
                acc = (not neg) and (not sub) and any(ts >= t and ev >= e for t, e in P["dup"])
            else:
                ev, det = evidence(a, b)
                acc = decide(ts, ev, unknown, sub)
            det["subst"] = int(sub)
            rows.append((x, y, a["src"], b["src"], round(ts, 4), round(ev, 2), int(unknown), int(acc),
                         json.dumps(det, sort_keys=True)))
ps = pd.DataFrame(rows, columns=["n1", "n2", "s1", "s2", "ts", "ev", "unknown_platform", "accept", "detail"])
ps["score"] = (ps.ts + 0.1 * ps.ev).round(4)
ps.to_csv(f"{W}/state/pair_scores.csv", index=False)
print("scored pairs", len(ps), "accepted", int(ps.accept.sum()), round(time.time() - t0, 1), "s")
print(ps.groupby(["s1", "s2", "unknown_platform"]).accept.agg(["size", "sum"]))
json.dump(P, open(f"{W}/state/match_params.json", "w"))
