"""Stage 6: attribute-wise fusion from each cluster's own member records."""
import pandas as pd, numpy as np, sys, json, re, collections
sys.path.insert(0,'work')
from norm import *
N = pd.read_pickle('work/state/clustered.pkl')
PRIO = {'metacritic':0,'sales':1,'dbpedia':2}
# canonical display for dbpedia-only platform keys: most frequent raw dbpedia spelling
dbdisp = N[N.source=='dbpedia'].groupby('pkey').platform_raw.agg(lambda s: s.value_counts().index[0]).to_dict()
def plat_display(r):
    if r.source == 'metacritic': return r.platform_raw
    if r.source == 'sales': return SALES_FIX.get(r.platform_raw, r.platform_raw)
    if r.pkey in DISPLAY: return DISPLAY[r.pkey]
    return dbdisp.get(r.pkey, r.platform_raw)
GENRE_RULES = [
 (r"action.?adventure", 'Action-Adventure'), (r"role.?playing|rpg|roguelike|dungeon", 'Role-Playing Game (RPG)'),
 (r"shoot|shooter|fighting|beat.?em|platform|hack and slash|action|fps|metroidvania|light gun", 'Action'),
 (r"adventure|point.and.click|visual novel|interactive (fiction|movie)|text", 'Adventure'),
 (r"horror", 'Horror'), (r"racing|driving|kart|rally|motocross|automobile", 'Racing'),
 (r"sport|soccer|football|basketball|baseball|golf|tennis|hockey|wrestling|boxing|skate|snowboard|olympic|bowling|billiards|cricket|rugby|fishing|hunting", 'Sports'),
 (r"strateg|tactic|4x|moba|tower defense|wargame|real.time|turn.based|city.building|artillery", 'Strategy'),
 (r"simulat|\bsim\b|tycoon|management|virtual life|flight|farming|life", 'Simulation'),
 (r"puzzle|logic|matching|stacking|hidden object", 'Puzzle'), (r"music|rhythm|danc|karaoke", 'Rhythm / Music'),
 (r"party|minigame|trivia|game show", 'Party / Social'), (r"card|board|pinball|parlor|gambling|casino", 'Card & Board'),
 (r"educ|edutainment", 'Educational'),
]
GENRE_RULES = [(re.compile(p, re.I), g) for p, g in GENRE_RULES]
def split_genres(r):
    g = r.genres_raw
    if not isinstance(g, str) or not g.strip(): return []
    return [x.strip() for x in g.split(',')] if r.source == 'metacritic' else [g.strip()]
def normkey(v): return re.sub(r"[^a-z0-9]+", ' ', fold(str(v))).strip()
def vote(vals):
    """vals: list of (value, source, order). Majority over normalized value, counting each source once;
    ties -> source priority (metacritic > sales > dbpedia), then first record order."""
    vals = [(v,s,o) for v,s,o in vals if v is not None and not (isinstance(v,float) and np.isnan(v)) and str(v).strip() != '']
    if not vals: return None, 0, 0
    per = collections.defaultdict(set); best = {}
    for v,s,o in vals:
        k = normkey(v) if isinstance(v,str) else v
        per[k].add(s)
        cand = (PRIO[s], o)
        if k not in best or cand < best[k][0]: best[k] = (cand, v)
    ranked = sorted(per, key=lambda k: (-len(per[k]), best[k][0]))
    return best[ranked[0]][1], len(per[ranked[0]]), len(per)
def mode_db(vals):
    vals = [v for v in vals if isinstance(v,str) and v.strip()]
    if not vals: return None
    c = collections.Counter(vals); m = max(c.values())
    return next(v for v in vals if c[v] == m)   # first occurrence among most frequent
out, prov = [], []
N['ordn'] = N.record_id.str.extract(r'_(\d+)$')[0].astype(int)
for cid, g in N.sort_values(['ordn']).groupby('cluster_id', sort=True):
    g = g.assign(prio=g.source.map(PRIO)).sort_values(['prio','ordn'])
    recs = list(g.itertuples())
    db = g[g.source=='dbpedia']; nondb = [r for r in recs if r.source != 'dbpedia']
    row = {'_id': cid, 'id': cid}
    # name: highest-priority source title (raw, dbpedia disambiguation stripped)
    row['name'] = nondb[0].name_raw.strip() if nondb else db.name.iloc[0]
    # platform: highest-priority source spelling
    row['platform'] = plat_display(recs[0])
    # per-source single values; dbpedia contributes its group mode (cross-product rows)
    def src_vals(col, dbval=None):
        v = [(getattr(r,col), r.source, r.ordn) for r in nondb]
        if len(db):
            v.append((dbval if dbval is not None else mode_db(db[col].tolist()), 'dbpedia', int(db.ordn.min())))
        return v
    # releaseYear: vote on YEAR; dbpedia uses disambiguation year else earliest year of its group
    dby = None
    if len(db):
        dy = db.disamb_year.dropna()
        dby = int(dy.iloc[0]) if len(dy) else (int(db.year.min()) if db.year.notna().any() else None)
    yv, ysup, yn = vote(src_vals('year', dby))
    row['releaseYear'] = f"{int(yv):04d}-01-01" if yv is not None and 1960 <= int(yv) <= 2024 else None
    dv, dsup, dn = vote(src_vals('developer'))
    row['developer'] = dv
    pv, _, _ = vote(src_vals('publisher', None) if not len(db) else [(r.publisher, r.source, r.ordn) for r in nondb])
    row['publisher'] = pv
    cv, _, _ = vote([(r.criticScore, r.source, r.ordn) for r in nondb])
    row['criticScore'] = int(round(cv)) if cv is not None else None
    uv, _, _ = vote([(r.userScore, r.source, r.ordn) for r in nondb])
    row['userScore'] = round(float(uv),1) if uv is not None else None
    ev, _, _ = vote([(r.ESRB, r.source, r.ordn) for r in nondb])
    row['ESRB'] = ev
    row['series'] = mode_db(db.series.tolist()) if len(db) else None
    # genres: union, source labels first, then taxonomy top-level names
    labels, seen = [], set()
    for r in recs:
        for x in split_genres(r):
            k = normkey(x)
            if k and k not in seen: seen.add(k); labels.append(x)
    tax = []
    for x in labels:
        for p, gname in GENRE_RULES:
            if p.search(x) and gname not in tax: tax.append(gname); break
    gl = labels + [t for t in tax if normkey(t) not in seen]
    if len(gl) > 10:  # keep taxonomy names, trim excess source labels (least-priority last)
        keep_t = [t for t in tax if normkey(t) not in seen]
        gl = labels[:10-len(keep_t)] + keep_t
    row['genres'] = json.dumps(gl, ensure_ascii=False) if gl else None
    out.append(row)
    prov.append(dict(_id=cid, n=len(g), sources='+'.join(sorted(set(g.source))), year_support=ysup, year_distinct=yn,
                     dev_support=dsup, dev_distinct=dn))
cols = ['_id','id','name','releaseYear','developer','genres','publisher','platform','criticScore','userScore','ESRB','series']
F = pd.DataFrame(out)[cols]
F['criticScore'] = F.criticScore.astype('Int64')
F.to_csv('submission/fused.csv', index=False)
pd.DataFrame(prov).to_csv('work/state/fusion_provenance.csv', index=False)
print(F.shape); print(F.notna().mean().round(3).to_dict())
