"""Stage 4+5: pairwise scoring and constrained clustering.
Score = text similarity + identifier evidence - contradictions (hand-set weights, no training).
Clustering exploits the observed source structure (ds1 and ds2 have identical per-type counts; ds3/ds4 are
smaller): at most one record per source per entity, built by successive one-to-one assignments
(Hungarian) ds1<->ds2, then ds3 -> clusters, then ds4 -> clusters. Assignments below ACCEPT stay singletons."""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

CFG = dict(ACCEPT=0.2, VETO=0.0, W_CODE=0.35, W_MPN=0.25, P_MPN=0.6, W_CAP=0.15, P_CAP=1.0, W_CHIP=0.1, P_CHIP=1.0,
           P_VRAM=0.6, P_BRAND=1.0, P_EXCL=0.4, P_MEM=0.3, W_VAR=0.03, P_VAR=0.04, W_MTOK=0.15, P_MTOK=0.4, P_RPM=0.5, P_SASG=0.4, P_IFACE=0.5, P_FF=1.0, SOFT_ID_FACTOR=0.3, W_LINE=0.05, P_LINE=0.1, ORDER=['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4'])
if len(sys.argv) > 1: CFG.update(json.loads(sys.argv[1]))

d = pd.read_pickle('state/s3_records.pkl')
c = pd.read_pickle('state/s3_candidates.pkl')
# soft (extraction-noise-prone) contradictions are down-weighted when an identifier matches exactly
soft = np.where((c.code > 0) | (c.mpn > 0), CFG['SOFT_ID_FACTOR'], 1.0)
c['score'] = (0.5 * c.sim_char + 0.5 * c.sim_word + CFG['W_CODE'] * c.code
              + np.where(c.mpn > 0, CFG['W_MPN'], np.where(c.mpn < 0, -CFG['P_MPN'], 0))
              + np.where(c.cap > 0, CFG['W_CAP'], np.where(c.cap < 0, -CFG['P_CAP'], 0))
              + np.where(c.chip > 0, CFG['W_CHIP'], np.where(c.chip < 0, -CFG['P_CHIP'], 0))
              - CFG['P_VRAM'] * (c.vram < 0) - CFG['P_BRAND'] * (c.brand < 0) - soft * CFG['P_MEM'] * (c.mem < 0)
              + CFG['W_VAR'] * c.var_common - CFG['P_VAR'] * c.var_diff
              + np.where(c.mtok > 0, CFG['W_MTOK'], np.where(c.mtok < 0, -soft * CFG['P_MTOK'], 0))
              - soft * (CFG['P_EXCL'] * c.excl + CFG['P_RPM'] * (c.rpm < 0) + CFG['P_SASG'] * (c.sasg < 0) + CFG['P_IFACE'] * (c.iface < 0) + CFG['P_FF'] * (c.ffk < 0))
              + CFG['W_LINE'] * c.line_common - CFG['P_LINE'] * c.line_diff)
c.to_pickle('state/s4_scored.pkl')

pair = {}
for i, j, s in zip(c.i.values, c.j.values, c.score.values):
    pair[(i, j)] = s; pair[(j, i)] = s

cluster_of = {}   # record index -> cluster key
members = {}      # cluster key -> list of record idx
log = []
order = CFG['ORDER']
for ptype, g in d.groupby('product_type'):
    # seed clusters from first source
    for i in g.index[g.source == order[0]]:
        cluster_of[i] = i; members[i] = [i]
    for src in order[1:]:
        new = list(g.index[g.source == src])
        cl = sorted({cluster_of[i] for i in g.index if i in cluster_of})
        M = np.full((len(new), len(cl)), -9.0)
        for a, i in enumerate(new):
            for b, k in enumerate(cl):
                ss = [pair[(i, m)] for m in members[k] if (i, m) in pair]
                if ss:
                    # cluster affinity: mean over members with candidate evidence, penalised by strongest contradiction
                    # a clearly contradicting member vetoes the cluster (prevents weak transitive bridges)
                    M[a, b] = 0.5 * np.mean(ss) + 0.5 * max(ss) if min(ss) >= CFG['VETO'] else min(ss)
        # dummy 'stay unassigned' column per record valued at the threshold: nobody is forced into a bad match
        Maug = np.hstack([M, np.full((len(new), len(new)), CFG['ACCEPT'] - 1e-6)])
        ra, cb = linear_sum_assignment(-Maug)
        assigned = set()
        for a, b in zip(ra, cb):
            if b >= len(cl):
                bb = int(np.argmax(M[a])) if len(cl) else 0
                log.append(dict(record=int(new[a]), cluster=int(cl[bb]) if len(cl) else -1, score=float(M[a, bb]) if len(cl) else -9, step=src, decision='unassigned_best_alt'))
                continue
            if M[a, b] >= CFG['ACCEPT']:
                k = cl[b]; i = new[a]
                cluster_of[i] = k; members[k].append(i); assigned.add(i)
                log.append(dict(record=int(i), cluster=int(k), score=float(M[a, b]), step=src, decision='assign'))
            else:
                log.append(dict(record=int(new[a]), cluster=int(cl[b]), score=float(M[a, b]), step=src, decision='reject_below_threshold'))
        for i in new:
            if i not in assigned:
                cluster_of[i] = i; members[i] = [i]

d['cluster_key'] = d.index.map(cluster_of)
d.to_pickle('state/s4_clustered.pkl')
pd.DataFrame(log).to_csv('state/s4_assign_log.csv', index=False)
json.dump(CFG, open('state/s4_config.json', 'w'), indent=1)
sz = d.groupby('cluster_key').size()
print('clusters', len(sz), 'size dist', sz.value_counts().sort_index().to_dict())
