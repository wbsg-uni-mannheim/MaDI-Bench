import math, re, numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from textnorm import name_key, ascii_fold
GENERIC_EXTRA = {'computer','computers','holdings','holding','group','industries','industry','industrial','corporation','company','co','international','the','and',
                 'plc','limited','technologies','technology','enterprises','motor','motors','systems','inc','corp','ltd','sa','ag','nv','se','financial','services',
                 'communications','entertainment','resources','brands','products','worldwide','global','bancorp','bancshares','foods','energy','pharmaceutical',
                 'pharmaceuticals','telecom','telecommunications','electronics','electric','electrical','manufacturing','mfg','laboratories','studios','partners','media',
                 'networks','network','software','solutions','insurance','bank','banking','properties','realty','capital','investments','investment','trust','of','de'}
def variants(r):
    """(key, weight) name variants. Authoritative identity names (dbpedia URI title, forbes URL slug) get weight 1;
    the raw name column gets weight 1 when consistent with the title, else 0.85 (alias / possibly swapped name)."""
    out = []
    t = r['tkey_core'] if isinstance(r['tkey_core'], str) else ''
    n = r['nkey_core'] if isinstance(r['nkey_core'], str) else ''
    if t:
        out.append((t, 1.0))
        if n and n != t:
            out.append((n, 1.0 if fuzz.token_set_ratio(n, t) >= 85 else 0.85))
    elif n:
        out.append((n, 1.0))
    return out
def ksim(a, b):
    if not a or not b: return 0.0
    ns_a, ns_b = a.replace(' ', ''), b.replace(' ', '')
    s = max(fuzz.token_sort_ratio(a, b), fuzz.ratio(ns_a, ns_b)) / 100
    s = max(s, JaroWinkler.similarity(ns_a, ns_b) - 0.05)
    return s
def extra_tokens(a, b):
    ta, tb = set(a.split()), set(b.split())
    inter = ta & tb
    if not inter: return None
    ea, eb = ta - tb, tb - ta
    ex = {x for x in ea | eb if x not in GENERIC_EXTRA}
    return ex
IDF = {}
def set_idf(keys):
    import collections
    df = collections.Counter()
    for k in keys:
        for t in set(k.split()): df[t] += 1
    N = max(len(keys), 1)
    IDF.clear(); IDF.update({t: math.log(N / c) for t, c in df.items()})
def idf(t): return IDF.get(t, math.log(len(IDF) + 1 or 2))
from rapidfuzz.distance import Levenshtein
def tok_sim(a, b):
    """token equality tolerant to OCR/typo noise: 0 edits for <=4 chars, 1 edit for 5-8, 2 edits for >=9"""
    if a == b: return 1.0
    L = min(len(a), len(b))
    allowed = 0 if L <= 4 else (1 if L <= 8 else 2)
    if allowed == 0: return 0.0
    d = Levenshtein.distance(a, b)
    if d <= allowed: return max(0.8, 1 - d / max(len(a), len(b)))
    return 0.0
def soft_cov(ka, kb):
    A, B = ka.split(), kb.split()
    if not A or not B: return 0.0, 0.0, set(), set()
    usedB = set(); wa = sum(idf(t) for t in A); wb = sum(idf(t) for t in B)
    ma = 0.0; mb = 0.0; unA = set(); 
    for t in sorted(A, key=lambda x: -idf(x)):
        best = (0.0, None)
        for j, u in enumerate(B):
            if j in usedB: continue
            s = tok_sim(t, u)
            if s > best[0]: best = (s, j)
        if best[1] is not None and best[0] >= 0.8:
            usedB.add(best[1]); ma += idf(t) * best[0]; mb += idf(B[best[1]]) * best[0]
        else: unA.add(t)
    unB = {B[j] for j in range(len(B)) if j not in usedB}
    return ma / wa, mb / wb, unA, unB
def name_sim(ka, kb):
    """(score, info): min of IDF-weighted soft token coverage of both names; joined-string ratio as alternative."""
    ca, cb, ua, ub = soft_cov(ka, kb)
    s = min(ca, cb)
    if len(ka.split()) != len(kb.split()):   # concatenation variants: 'Sidus HQ' ~ 'SidusHQ'
        ja, jb = ka.replace(' ', ''), kb.replace(' ', '')
        if ja == jb or (min(len(ja), len(jb)) >= 8 and Levenshtein.distance(ja, jb) <= 1): s = max(s, 0.97)
    return s, ca, cb, ua, ub
def name_evidence(ra, rb):
    best = (-1, None, None, None)
    for ka, wa in variants(ra):
        for kb, wb in variants(rb):
            s, ca, cb, ua, ub = name_sim(ka, kb)
            s *= min(wa, wb)
            if s > best[0]: best = (s, ka, kb, (ca, cb, ua, ub))
    s, ka, kb, info = best
    s = max(s, 0)
    contain = 0.0
    if ka and kb:
        ex = extra_tokens(ka, kb)
        if ex is not None:
            ta, tb = set(ka.split()), set(kb.split())
            if (ta <= tb or tb <= ta):
                contain = 1.0 if not ex else 0.5
    return s, contain, ka, kb, info
FACT = [1.0, 0.7842, 0.9215, 1/0.7842, 1/0.9215, 0.7842/0.9215, 0.9215/0.7842]
def money_rel(a, b):
    """1 agree (incl. known perturbation factors), 0 unknown, -1 disagree"""
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)) or a <= 0 or b <= 0: return 0
    r = a / b
    for f in FACT:
        for sc in (1, 1e3, 1e-3):
            if abs(r / (f * sc) - 1) <= 0.02: return 1
    return -1
def nz(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))
def pair_features(ra, rb):
    ns, contain, ka, kb, info = name_evidence(ra, rb)
    f = {'info_a': sum(idf(t) for t in ka.split()) if ka else 0, 'info_b': sum(idf(t) for t in kb.split()) if kb else 0,'ns': ns, 'contain': contain, 'covmax': max(info[0], info[1]) if info else 0,
         'extra': ' '.join(sorted((info[2] | info[3]) - GENERIC_EXTRA)) if info else '', 'ka': ka, 'kb': kb}
    ca, cb = ra['country'], rb['country']
    f['country'] = 0 if not (nz(ca) and nz(cb)) else (1 if ca == cb else -1)
    ya, yb = ra['founded_year'], rb['founded_year']
    f['year'] = 0 if not (nz(ya) and nz(yb)) else (1 if abs(ya - yb) <= 1 else -1)
    xa, xb = ra['city'], rb['city']
    if nz(xa) and nz(xb):
        a_, b_ = ascii_fold(xa).lower(), ascii_fold(xb).lower()
        f['city'] = 1 if (fuzz.token_set_ratio(a_, b_) >= 85 or fuzz.partial_ratio(a_, b_) >= 90) else -1
    else: f['city'] = 0
    ia, ib = ra['industry'], rb['industry']
    f['industry'] = 0 if not (nz(ia) and nz(ib)) else (1 if ia == ib else -1)
    f['assets'] = money_rel(ra['assets'], rb['assets'])
    f['revenue'] = money_rel(ra['revenue'], rb['revenue'])
    pa, pb = ra['people'], rb['people']
    if pa and pb:
        A = {ascii_fold(x).lower() for x in pa}; B = {ascii_fold(x).lower() for x in pb}
        hit = any(fuzz.token_set_ratio(x, y) >= 85 for x in A for y in B)
        f['people'] = 1 if hit else -1
    else: f['people'] = 0
    return f
