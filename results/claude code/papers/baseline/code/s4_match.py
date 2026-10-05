"""Stage 4b: rule-based match decisions on candidate features (no training, no labels)."""
import os, re, json, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
N = pd.read_pickle('work/state/s2_normalized.pkl')
C = pd.read_pickle('work/state/s4_features.pkl')
src = N.source.to_numpy(); tk = N.tkey.to_numpy(); tt = N.ttok.to_numpy()
def vnum(v):
    m = re.match(r'\d+', v) if isinstance(v, str) else None
    return m.group(0) if m else None
vn = N.volume.map(vnum).to_numpy()
fp = N.first_page.to_numpy(); lp = N.last_page.to_numpy()
digs = [frozenset(t for t in x if t.isdigit()) for x in tt]
I, J = C.i.to_numpy(), C.j.to_numpy()
C['vconf'] = [vn[i] is not None and vn[j] is not None and vn[i] != vn[j] for i, j in zip(I, J)]
def pdiff(a, b): return a is not None and b is not None and str(a).lower() != str(b).lower()
# DBLP pages are article-local (always start at 1: '5:1-5:12' parsed as 1-12) -> never used as contradiction.
def isnum(x): return isinstance(x, str) and x.isdigit()
C['pconf'] = [src[i] != 'dblp' and src[j] != 'dblp' and all(isnum(v) for v in (fp[i], fp[j], lp[i], lp[j]))
              and pdiff(fp[i], fp[j]) and pdiff(lp[i], lp[j]) for i, j in zip(I, J)]
C['dconf'] = [(digs[i] ^ digs[j]) != frozenset() and not p for i, j, p in zip(I, J, C.prefix)]
# damaged/short title on one side: shorter key is a prefix (possibly empty) of the other
C['short_prefix'] = [(min(len(tk[i]), len(tk[j])) < 12) and (tk[j].startswith(tk[i]) or tk[i].startswith(tk[j])) for i, j in zip(I, J)]
ty = N.type.to_numpy()
# DBLP says journal article while Crossref says proceedings paper -> conference vs journal version.
# (the reverse, Crossref 'article' vs DBLP 'inproceedings', is systematic for AAAI/Procedia and not a conflict)
C['tconf'] = [(src[i] == 'crossref' and src[j] == 'dblp' and ty[i] == 'inproceedings' and ty[j] == 'article') for i, j in zip(I, J)]
vol_ok = C.vol_eq.eq(1) | C.fp_eq.eq(1)
C['hard_reject'] = C.tconf | C.vconf | C.pconf | C.dconf | (C.ydiff >= 3) | ((C.ydiff == 2) & ~vol_ok)
a = C.asim_max.fillna(-1); amin = C.asim_min.fillna(-1)
auth_missing = C.asim_max.isna()
auth_ok = (a >= 0.5) | auth_missing | (amin >= 0.8)
tr = C.tratio.fillna(0)
rules = {
 'A_title_hi': (tr >= 0.95) & (C.tlen_min >= 3) & (auth_ok | ((a > 0) & (C.ydiff == 0)) | vol_ok),
 'B_short_title': (tr >= 0.95) & (C.tlen_min < 3) & (a >= 0.5) & (C.ydiff <= 1),
 'C_title_mid': (tr >= 0.8) & (tr < 0.95) & (a >= 0.8) & (C.ydiff <= 1) & ((C.tcont >= 0.9) | vol_ok),
 'D_prefix': C.prefix & (a >= 0.8) & (C.ydiff <= 1),
 'E_damaged': C.short_prefix & (((a >= 0.9) & (C.n_auth_min >= 2) & (C.ydiff <= 1)) | ((a >= 0.99) & C.fp_eq.eq(1) & (C.ydiff == 0))),
}
C['rule'] = ''
for k, m in reversed(list(rules.items())):
    C.loc[m, 'rule'] = k
C['accept'] = (C.rule != '') & ~C.hard_reject
C['score'] = (0.6 * tr + 0.4 * a.clip(lower=0) + 0.05 * C.vol_eq.fillna(0) + 0.05 * C.fp_eq.fillna(0)
              - 0.05 * C.ydiff).round(4)
C['score'] = C.score.clip(0, 1)
C.to_pickle('work/state/s4_decisions.pkl')
diag = dict(stage='matching', accepted=int(C.accept.sum()),
            by_rule=C[C.accept].rule.value_counts().to_dict(),
            rule_but_rejected=C[(C.rule != '') & C.hard_reject].rule.value_counts().to_dict(),
            reject_reasons={k: int((C[k] & (C.rule != '')).sum()) for k in ['tconf','vconf','pconf','dconf']})
print(json.dumps(diag, indent=1)); json.dump(diag, open('work/state/s4_diag.json','w'))
