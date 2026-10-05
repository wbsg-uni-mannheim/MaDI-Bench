"""Stage 6: attribute-wise fusion from each cluster's own member records, with per-cell provenance."""
import pandas as pd, numpy as np, os, sys, json, re
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'work')
from common import basic, display_name
df = pd.read_pickle('work/state/norm.pkl').set_index('id')
memb = pd.read_csv('work/state/membership.csv')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
ATTRS = [a for a in schema['properties'] if a != 'id']
# source priority per attribute (deterministic tie-break after normalized vote); rationale in report.md
PRIO = {
 'name': ['forbes', 'dbpedia', 'fullcontact'],       # Forbes/DBpedia give common names; FullContact names are noisy
 'founded': ['dbpedia', 'fullcontact'],
 'country': ['forbes', 'dbpedia', 'fullcontact'],
 'city': ['fullcontact', 'dbpedia'],                 # FullContact city is a clean field; DBpedia city is parsed from a concatenated string
 'industry': ['forbes', 'dbpedia'],                  # Forbes segment -> GICS table is more reliable than keyword rules
 'assets': ['forbes', 'dbpedia'],                    # Forbes in US$; DBpedia unit/currency uncertain
 'revenue': ['forbes', 'dbpedia'],
 'keypeople': ['dbpedia', 'fullcontact'],            # DBpedia field = founders; FullContact = key persons (partly founders)
}
def clean_name(n, src):
    n = str(n)
    if src == 'dbpedia': n = re.sub(r'\s*\([^)]*\)\s*$', '', n)   # Wikipedia disambiguation suffix
    return display_name(n)
def value(rec, a):
    if a == 'name': return clean_name(rec['name'], rec.source)
    if a == 'founded': return None if pd.isna(rec.founded) else f'{int(rec.founded):04d}-01-01'
    if a in ('assets', 'revenue'): return None if pd.isna(rec[a]) else int(round(rec[a]))
    if a == 'keypeople': return None if rec.keypeople is None or pd.isna(rec.keypeople) else rec.keypeople
    v = rec[a]
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
def vkey(a, v):
    if a in ('name', 'city'): return basic(v)
    if a == 'founded': return v[:4]
    if a == 'keypeople': return tuple(sorted(basic(x) for x in json.loads(v)))
    if a in ('assets', 'revenue'): return round(np.log10(max(v, 1)), 2)
    return v
rows, prov = [], []
for cid, g in memb.groupby('cluster_id', sort=True):
    recs = [df.loc[i] for i in sorted(g.record_id)]
    out = {'_id': cid, 'id': cid}
    for a in ATTRS:
        cands = [(r.source, r.name, value(r, a)) for r in recs]
        cands = [c for c in cands if c[2] is not None and c[2] != '']
        cands = [c for c in cands if c[0] in PRIO[a]]
        if not cands: out[a] = None; continue
        cnt = Counter(vkey(a, c[2]) for c in cands)
        best = max(cands, key=lambda c: (cnt[vkey(a, c[2])], -PRIO[a].index(c[0]), str(c[1])))
        # a single source's duplicate rows count once only in spirit: votes are only decisive when >=2 distinct sources agree
        srcs_agree = {c[0] for c in cands if vkey(a, c[2]) == vkey(a, best[2])}
        if len(srcs_agree) < 2:
            best = min(cands, key=lambda c: (PRIO[a].index(c[0]), str(c[1])))
            rule = 'priority'
        else: rule = 'vote'
        out[a] = best[2]
        prov.append((cid, a, best[1], best[0], rule, len({vkey(a, c[2]) for c in cands})))
    rows.append(out)
fused = pd.DataFrame(rows)
fused['keypeople'] = fused.keypeople.where(fused.keypeople.notna(), None)
for c in ['assets', 'revenue']: fused[c] = fused[c].astype('Int64')
cols = ['_id'] + list(schema['properties'])
fused[cols].to_csv('work/state/fused.csv', index=False)
pd.DataFrame(prov, columns=['cluster_id', 'attribute', 'record_id', 'source', 'rule', 'n_distinct_values']).to_csv('work/state/fusion_provenance.csv', index=False)
print(fused[cols].notna().mean().round(3).to_dict())
