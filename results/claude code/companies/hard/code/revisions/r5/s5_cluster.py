"""Stage 5: graph clustering of accepted pairs with identity constraints.
Constraint: a cluster may not contain two *native* (non-injected) records of dbpedia or forbes (distinct URIs = distinct entities).
Violating components are split by removing weakest edges (lowest score first) until constraints hold."""
import os, sys, json, time, collections
import pandas as pd, numpy as np, networkx as nx
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def violates(nodes, R):
    c = collections.Counter()
    for n in nodes:
        if R.source.iat[n] in ('dbpedia', 'forbes') and not R.native_dup.iat[n]: c[R.source.iat[n]] += 1
    return any(v > 1 for v in c.values())
def main():
    R = pd.read_pickle(f'{ROOT}/work/state/records.pkl')
    P = pd.read_pickle(f'{ROOT}/work/state/pair_decisions.pkl')
    A = P[P.accept]
    G = nx.Graph(); G.add_nodes_from(range(len(R)))
    for r in A.itertuples(): G.add_edge(r.i, r.j, score=r.score, rule=r.rule)
    removed = []
    # constraint repair: separate each pair of conflicting natives by a minimum-weight edge cut (capacity = edge score)
    changed = True
    while changed:
        changed = False
        for comp in list(nx.connected_components(G)):
            if len(comp) < 2 or not violates(comp, R): continue
            nat = collections.defaultdict(list)
            for n in sorted(comp):
                if R.source.iat[n] in ('dbpedia', 'forbes') and not R.native_dup.iat[n]: nat[R.source.iat[n]].append(n)
            src_ = next(s_ for s_, v in sorted(nat.items()) if len(v) > 1)
            a, b = nat[src_][0], nat[src_][1]
            sub = G.subgraph(comp).copy()
            for u, v, d in sub.edges(data=True): d['capacity'] = max(d['score'], 0.01) + (10 if d['rule'] == 'forbes_url_link' else 0)
            cut_value, (S, T) = nx.minimum_cut(sub, a, b)
            cut = [(u, v) for u in S for v in sub.neighbors(u) if v in T]
            for u, v in cut:
                d = G.edges[u, v]
                G.remove_edge(u, v); removed.append((R.rid.iat[u], R.rid.iat[v], d['score'], d['rule'] + ':native_conflict_cut'))
            changed = True
    # conflict refinement: detach a member that strongly contradicts (>=2 of year/city/people) other members of its component
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import pair_features, set_idf
    set_idf(list(R.nkey_core) + [t for t in R.tkey_core if t])
    recs = R.to_dict('records')
    def strong_conf(a, b):
        f = pair_features(recs[a], recs[b])
        return (f['year'] == -1) + (f['city'] == -1) + (f['people'] == -1)
    detached = []
    changed = True
    while changed:
        changed = False
        for comp in list(nx.connected_components(G)):
            if len(comp) < 3: continue
            comp = sorted(comp)
            bad = {n: sum(1 for m in comp if m != n and strong_conf(n, m) >= 2) for n in comp}
            worst = max(comp, key=lambda n: (bad[n], -G.degree(n), n))
            if bad[worst] >= 2:
                for nb in list(G.neighbors(worst)):
                    d = G.edges[worst, nb]
                    G.remove_edge(worst, nb); removed.append((R.rid.iat[worst], R.rid.iat[nb], d['score'], d['rule'] + ':conflict_detach'))
                detached.append(R.rid.iat[worst]); changed = True
    comps = sorted(nx.connected_components(G), key=lambda c: min(c))
    cid = np.empty(len(R), dtype=object)
    for k, comp in enumerate(comps):
        for n in comp: cid[n] = f'E{k:05d}'
    R['cluster'] = cid
    R.to_pickle(f'{ROOT}/work/state/records_clustered.pkl')
    pd.DataFrame(removed, columns=['id1', 'id2', 'score', 'rule']).to_csv(f'{ROOT}/work/state/rejected_edges.csv', index=False)
    # final edge list = accepted edges still in graph
    kept = [(R.rid.iat[a], R.rid.iat[b], d['score'], d['rule']) for a, b, d in G.edges(data=True)]
    pd.DataFrame(kept, columns=['id1', 'id2', 'score', 'rule']).to_pickle(f'{ROOT}/work/state/kept_edges.pkl')
    sizes = R.groupby('cluster').size()
    nsrc = R.groupby('cluster').source.nunique()
    diag = {'ts': time.time(), 'stage': 'clustering', 'inputs': 'work/state/pair_decisions.pkl', 'records': len(R), 'clusters': int(len(sizes)),
            'singletons': int((sizes == 1).sum()), 'singleton_share': float((sizes == 1).mean()), 'max_size': int(sizes.max()),
            'size_dist': {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
            'sources_per_cluster': {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()},
            'edges_accepted': int(len(A)), 'edges_removed_by_constraints': len(removed), 'detached_for_conflicts': len(detached)}
    print(json.dumps(diag, indent=1))
    with open(f'{ROOT}/work/diagnostics.jsonl', 'a') as fh: fh.write(json.dumps(diag) + '\n')
if __name__ == '__main__':
    main()
