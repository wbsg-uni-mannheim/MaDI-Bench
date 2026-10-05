"""Stage 2: normalization. Writes work/state/norm.pkl (all records, raw + canonical columns)."""
import os, re, json, unicodedata
from collections import Counter, defaultdict
import pandas as pd, numpy as np
from datetime import datetime
from platforms import canon_platform, pkey
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRC = ["dbpedia", "metacritic", "sales"]

def ws(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    x = re.sub(r'\s+', ' ', str(x)).strip()
    return x or None
def fold(x):
    x = unicodedata.normalize('NFKD', x); return ''.join(c for c in x if not unicodedata.combining(c))
def key(x):
    if x is None: return None
    k = re.sub(r'[^a-z0-9]', '', fold(x).lower().replace('&', 'and'))
    return k or None
MONTHS = {m: i for i, m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"], 1)}
def parse_date(x):
    x = ws(x)
    if not x: return None
    m = re.match(r'^([A-Za-z]+)\s*(\d{1,2})\s*,\s*(\d[\d ]*)$', x)
    if m and m.group(1).lower() in MONTHS:
        y = re.sub(r'\D', '', m.group(3))
        if len(y) == 4: return f"{y}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    dg = re.sub(r'\D', '', x)
    if len(dg) == 8:
        try: datetime.strptime(dg, "%Y%m%d"); return f"{dg[:4]}-{dg[4:6]}-{dg[6:]}"
        except ValueError: return None
    if len(dg) == 4: return f"{dg}-01-01"
    return None
def num(x):
    x = ws(x)
    if not x: return None
    try: return float(x.replace(' ', ''))
    except ValueError: return None
ESRB_OK = {"E","E10+","T","M","AO","RP","RP-LM17"}
def esrb(x):
    x = ws(x)
    if not x: return None
    x = x.upper().replace(' ', '')
    x = {"K-A": "E", "KA": "E", "EC": None}.get(x, x)
    return x if x in ESRB_OK else None
PAREN = re.compile(r'\s*\((?:[^()]*\b(?:company|developer|studio|video game|software|franchise|series|publisher|game|corporation|card game|board game)\b[^()]*)\)\s*$', re.I)
def strip_disamb(x):
    x = ws(x)
    return ws(PAREN.sub('', x)) if x else None

frames = []
for s in SRC:
    d = pd.read_csv(f"{ROOT}/task/input/data/{s}.csv", dtype=str, keep_default_na=False, na_values=[""])
    for c in ["name","releaseYear","developer","publisher","platform","genres","series","criticScore","userScore","ESRB"]:
        if c not in d: d[c] = None
    d["source"] = s
    frames.append(d[["id","source","name","releaseYear","developer","publisher","platform","genres","series","criticScore","userScore","ESRB"]])
R = pd.concat(frames, ignore_index=True)
for c in ["name","developer","publisher","platform","genres","series"]: R[c] = R[c].map(ws)

# platform: alias map; for unmapped keys use most frequent raw spelling of that key
pk_spell = defaultdict(Counter)
for v in R.platform.dropna(): pk_spell[pkey(v)][v] += 1
fb = {k: c.most_common(1)[0][0] for k, c in pk_spell.items()}
R["platform_c"] = R.platform.map(lambda v: canon_platform(v, fb))
R["platform_k"] = R.platform_c.map(lambda v: pkey(v) if v else None)

# name: key + global preferred spelling (most frequent raw form per key; tie -> fewest odd case flips)
R["name_k"] = R.name.map(key)
def best_spelling(counter):
    return sorted(counter.items(), key=lambda kv: (-kv[1], sum(1 for a, b in zip(kv[0], kv[0][1:]) if a.islower() and b.isupper()), -kv[0].count(' '), kv[0]))[0][0]
def spelling_map(series):
    sp = defaultdict(Counter)
    for v in series.dropna(): sp[key(v)][v] += 1
    return {k: best_spelling(c) for k, c in sp.items()}
NAME_SP = spelling_map(R.name)
R["name_c"] = R.name_k.map(lambda k: NAME_SP.get(k) if k else None)

R["date_c"] = R.releaseYear.map(parse_date)
R["year"] = R.date_c.map(lambda d: int(d[:4]) if d else None)
R["critic_c"] = R.criticScore.map(num)
R["user_c"] = R.userScore.map(num)
R.loc[~R.critic_c.between(0, 100), "critic_c"] = None
R.loc[~R.user_c.between(0, 10), "user_c"] = None
R["esrb_c"] = R.ESRB.map(esrb)

# developer: dbpedia disambiguation suffix removed; values that are actually platform names dropped
PLAT_KEYS = set(k for k in pk_spell) | {pkey(v) for v in R.platform_c.dropna()}
def dev_list(v, s):
    if not v: return []
    parts = [v] if s == "dbpedia" else [p for p in re.split(r'\s*,\s*', v)]
    out = []
    for p in parts:
        p = strip_disamb(p)
        if not p or key(p) is None: continue
        if s == "dbpedia" and key(p) in PLAT_KEYS: continue
        out.append(p)
    return out
R["dev_list"] = [dev_list(v, s) for v, s in zip(R.developer, R.source)]
DEV_SP = spelling_map(pd.Series([p for l in R.dev_list for p in l]))
R["dev_list"] = R.dev_list.map(lambda l: [DEV_SP[key(p)] for p in l])
R["pub_c"] = R.publisher.map(ws)
R["series_c"] = R.series.map(strip_disamb)
SER_SP = spelling_map(R.series_c)
R["series_c"] = R.series_c.map(lambda v: SER_SP.get(key(v)) if v else None)

# genres: split lists; repair token-scrambled values by matching token multiset to known clean genre strings
def gsplit(v, s):
    if not v: return []
    return [ws(p) for p in v.split(',') if ws(p)]
R["genre_raw"] = [gsplit(v, s) for v, s in zip(R.genres, R.source)]
allg = Counter(g for l in R.genre_raw for g in l)
bag = lambda g: tuple(sorted(re.findall(r"[a-z0-9']+|-", g.lower())))
bag_best = defaultdict(Counter)
for g, c in allg.items(): bag_best[bag(g)][g] += c
# prefer spellings starting uppercase and with natural order (most frequent)
def bestg(c):
    return sorted(c.items(), key=lambda kv: (not kv[0][:1].isupper(), -kv[1], kv[0]))[0][0]
BAG = {b: bestg(c) for b, c in bag_best.items()}
G_SP = spelling_map(pd.Series(list(allg.elements())))
def gfix(g):
    b = BAG.get(bag(g), g)
    return G_SP.get(key(b), b)
R["genre_list"] = R.genre_raw.map(lambda l: list(dict.fromkeys(gfix(g) for g in l)))

# junk flag: sales rows with no name and no platform (developer + scrambled genre only)
R["junk"] = R.name_k.isna()
R["rid"] = R.id
# dbpedia collection/list/series articles describe several distinct games under one title -> not groupable by title
COLL = re.compile(r'^list of |\bvideo game series\b|\bvideo game collections?\b|\bgames series\b', re.I)
R["collection_page"] = (R.source == "dbpedia") & R.name.fillna("").str.contains(COLL)
R["unit"] = R.source + "|" + R.name_k.fillna("") + "|" + R.platform_k.fillna("")
R.loc[R.junk, "unit"] = "junk|" + R.rid[R.junk]
R.loc[R.collection_page & ~R.junk, "unit"] = "row|" + R.rid[R.collection_page & ~R.junk]
print("collection-page rows", int(R.collection_page.sum()))
os.makedirs(f"{W}/state", exist_ok=True)
R.to_pickle(f"{W}/state/norm.pkl")
# taxonomy coverage
tax_p = set(pd.read_csv(f"{ROOT}/task/input/schemamatching/Gaming_Platforms_Taxonomy.csv")["Platform Name"])
rows = []
for s in SRC:
    x = R[R.source == s]
    nn = x.platform_c.notna().sum(); cn = x.platform_c.isin(tax_p).sum()
    rows.append(dict(source=s, attribute="platform", non_null=int(nn), canonical=int(cn), unmapped=int(nn-cn), canonical_rate=round(cn/max(nn,1),4)))
    nn = x.esrb_c.notna().sum(); rows.append(dict(source=s, attribute="ESRB", non_null=int(x.ESRB.notna().sum()), canonical=int(nn), unmapped=int(x.ESRB.notna().sum()-nn), canonical_rate=round(nn/max(x.ESRB.notna().sum(),1),4)))
    rows.append(dict(source=s, attribute="releaseYear", non_null=int(x.releaseYear.notna().sum()), canonical=int(x.date_c.notna().sum()), unmapped=int(x.releaseYear.notna().sum()-x.date_c.notna().sum()), canonical_rate=round(x.date_c.notna().sum()/max(x.releaseYear.notna().sum(),1),4)))
pd.DataFrame(rows).to_csv(f"{W}/taxonomy_coverage.csv", index=False)
print(pd.DataFrame(rows).to_string())
print(R.groupby("source").size())
