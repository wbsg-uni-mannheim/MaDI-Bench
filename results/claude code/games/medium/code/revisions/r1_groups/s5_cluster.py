"""Stage 5: refinement / clustering.
Phase 1: greedy union of accepted same-platform node edges (descending score) under constraints
         <=1 metacritic record, <=1 sales record per cluster; dbpedia groups with conflicting disambiguation years never merge.
Phase 2: records without a usable platform attach to a platform cluster only when unambiguous (see report)."""
import pandas as pd, numpy as np, os, collections, json
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
U = pd.read_pickle(f'{BASE}/work/state/s4_nodes.pkl')
P = pd.read_pickle(f'{BASE}/work/state/s4_pairs.pkl')
acc = P[P.reject.isna()].copy()
node_src = dict(zip(U.node, U.source))
node_pk = U.groupby('node').pkb.first().to_dict()
node_dy = U.groupby('node').dyear.first().to_dict()
node_nkey = U.groupby('node').nkey.first().to_dict()
nodes = sorted(U.node.unique())
parent = {n: n for n in nodes}
members = {n: [n] for n in nodes}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def compatible(ra, rb):
    ma, mb = members[ra], members[rb]
    cnt = collections.Counter(node_src[n] for n in ma + mb)
    if cnt['metacritic'] > 1 or cnt['sales'] > 1: return False
    dys = {node_dy[n] for n in ma + mb if node_src[n] == 'dbpedia' and node_dy[n] == node_dy[n]}
    if len(dys) > 1: return False
    pks = {node_pk[n] for n in ma + mb if node_pk[n] is not None and node_pk[n] == node_pk[n]}
    return len(pks) <= 1
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb or not compatible(ra, rb): return False
    if len(members[ra]) < len(members[rb]): ra, rb = rb, ra
    parent[rb] = ra; members[ra] += members.pop(rb); return True
# node-level edges (max score over record pairs)
E = acc.groupby(['na', 'nb']).agg(score=('score', 'max'), rule=('rule', 'min'), ev=('evidence', lambda s: '|'.join(sorted(set('|'.join(s).split('|')) - {''})))).reset_index()
E['pa'] = E.na.map(node_pk); E['pb'] = E.nb.map(node_pk)
E1 = E[E.pa.notna() & E.pb.notna() & (E.pa == E.pb)].sort_values(['score', 'na', 'nb'], ascending=[False, True, True])
rejected = []
for na, nb, sc in E1[['na', 'nb', 'score']].itertuples(index=False):
    if not union(na, nb) and find(na) != find(nb): rejected.append((na, nb, sc, 'constraint'))
print('phase1 edges', len(E1), 'rejected by constraint', len(rejected))
# ---------- phase 2: nodes without platform ----------
miss = [n for n in nodes if node_pk[n] is None or node_pk[n] != node_pk[n]]
# platforms known for each dbpedia title (same normalized title + same disambiguation year), to detect multi-platform games
dtitle_pks = collections.defaultdict(set)
for n in nodes:
    if node_src[n] == 'dbpedia' and isinstance(node_pk[n], str) and node_nkey[n]:
        dtitle_pks[(node_nkey[n], node_dy[n] if node_dy[n] == node_dy[n] else None)].add(node_pk[n])
E2 = E[E.na.isin(miss) | E.nb.isin(miss)]
attached = 0; phase2_log = []
missset = set(miss)
adj = collections.defaultdict(list)
for na, nb, sc, ev in E2[['na', 'nb', 'score', 'ev']].itertuples(index=False):
    if na in missset: adj[na].append((nb, sc, ev))
    if nb in missset: adj[nb].append((na, sc, ev))
for n in sorted(adj):
    opts = collections.defaultdict(lambda: (0, ''))
    for o, sc, ev in adj[n]:
        if o in missset: continue
        r = find(o)
        if sc > opts[r][0]: opts[r] = (sc, ev)
    pks = {node_pk[members[r][0]] for r in opts} | {node_pk[m] for r in opts for m in members[r] if isinstance(node_pk[m], str)}
    pks = {p for p in pks if isinstance(p, str)}
    if node_src[n] == 'dbpedia':
        pks |= dtitle_pks.get((node_nkey[n], node_dy[n] if node_dy[n] == node_dy[n] else None), set())
    elig = {r: v for r, v in opts.items() if compatible(find(n), r)}
    choice, why = None, ''
    strong = {r: v for r, v in elig.items() if 'critic=' in v[1]}
    if len(strong) == 1:
        choice, why = next(iter(strong)), 'unique critic-score agreement'
    elif len(pks) == 1 and len(elig) == 1 and len(opts) == 1:
        choice, why = next(iter(elig)), 'title known on a single platform'
    if choice is not None and union(n, choice):
        attached += 1; phase2_log.append((n, choice, why))
print('missing-platform nodes', len(miss), 'with candidates', len(adj), 'attached', attached)
# ---------- export cluster assignment ----------
PRIO = {'metacritic': 0, 'sales': 1, 'dbpedia': 2}
U['root'] = U.node.map(find)
first = U.sort_values(['source'], key=lambda s: s.map(PRIO)).groupby('root').rid.first()
# deterministic cluster id: 'e_' + id of highest-priority first member (metacritic > sales > dbpedia, then id order)
U['_ord'] = U.source.map(PRIO)
first = U.sort_values(['_ord', 'rid']).groupby('root').rid.first()
U['cluster_id'] = 'e_' + U.root.map(first)
U.drop(columns=['_ord']).to_pickle(f'{BASE}/work/state/s5_clusters.pkl')
pd.DataFrame(rejected, columns=['na', 'nb', 'score', 'why']).to_csv(f'{BASE}/work/state/s5_rejected_edges.csv', index=False)
pd.DataFrame(phase2_log, columns=['node', 'cluster_root', 'why']).to_csv(f'{BASE}/work/state/s5_phase2_attach.csv', index=False)
cs = U.groupby('cluster_id').agg(n=('rid', 'size'), srcs=('source', lambda s: '+'.join(sorted(set(s)))))
print('clusters', len(cs)); print(cs.srcs.value_counts().to_dict()); print(cs.n.describe().to_dict())
