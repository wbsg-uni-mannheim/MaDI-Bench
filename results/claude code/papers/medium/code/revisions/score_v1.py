import numpy as np, pandas as pd
def score(f):
    """Additive evidence score (hand-set log-odds-like weights, no training)."""
    ts = np.maximum(f.tsim.fillna(0).values, f.tsort.fillna(0).values)
    s = np.zeros(len(f))
    # title
    s += np.select([ts >= 0.97, ts >= 0.90, ts >= 0.80, ts >= 0.65, ts >= 0.5], [4.0, 3.0, 1.5, -1.0, -2.5], -4.0)
    trunc = (f.prefix.values == 1) & (ts < 0.97)
    # truncated title: prefix of the other title; worth more when the prefix is long
    s = np.where(trunc, np.maximum(s, np.where(f.tlen_s.values >= 25, 3.0, np.where(f.tlen_s.values >= 10, 2.0, 1.0))), s)
    # main title (text before ':' etc.) equals the other title: typical Crossref/OpenAlex vs DBLP difference
    s = np.where((f.main.values == 1) & (ts < 0.97), np.maximum(s, 3.5), s)
    # authors
    a = np.fmax(np.clip(f.asim.values, 0, 1), f.atok.values)  # token-bag overlap rescues scrambled author lists
    s += np.where(np.isnan(a), 0, np.select([a >= 0.8, a >= 0.5, a >= 0.3], [2.0, 1.0, 0.0], -2.5))
    # year
    y = f.ydiff.values
    s += np.where(np.isnan(y), 0, np.select([y == 0, y == 1], [0.5, -0.5], -2.0))
    # venue
    v = f.venue.values
    s += np.where(np.isnan(v), 0, np.select([v >= 0.99, v >= 0.6], [1.0, 0.0], -1.5))
    # volume / pages
    # agreement on uninformative values (page '1' of article-numbered journals, volume == year) earns little
    weak = {'vol_eq': f.vol_same.str.fullmatch(r'20[12]\d').values, 'fp_eq': (f.fp_same == '1').values}
    for col, pos, neg in (('vol_eq', 1.5, -2.5), ('fp_eq', 1.5, -2.0), ('lp_eq', 0.5, -0.5), ('iss_eq', 0.5, -0.5)):
        e = f[col].values
        p = np.where(weak[col], 0.3, pos) if col in weak else pos
        s += np.where(np.isnan(e), 0, np.where(e == 1, p, neg))
    # type conflict: article vs inproceedings (only crossref and dblp distinguish these)
    t1, t2 = f.type1.values, f.type2.values
    conf = (((t1 == 'article') & (t2 == 'inproceedings')) | ((t1 == 'inproceedings') & (t2 == 'article')))
    both_cd = f.id1.str.startswith('crossref').values & f.id2.str.startswith('dblp').values
    s += np.where(conf & both_cd, -1.0, 0)
    return s
