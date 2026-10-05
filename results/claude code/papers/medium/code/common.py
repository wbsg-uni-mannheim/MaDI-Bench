import numpy as np
from rapidfuzz import fuzz
STOP = {'of','the','and','on','in','for','a','an','de','la','und','&'}
def jtoks(k): return [t for t in k.split() if t not in STOP]
def venue_compat(a, b):
    """1 if one venue key is an abbreviation / token-set equal of the other, 0 if both present but unrelated, nan if missing."""
    if not a or not b: return np.nan
    if a == b: return 1.0
    ta, tb = jtoks(a), jtoks(b)
    if set(ta) == set(tb): return 1.0
    for s, l in ((ta, tb), (tb, ta)):   # abbreviation: each short token prefix of a long token, in order
        i = 0
        for t in s:
            while i < len(l) and not l[i].startswith(t): i += 1
            if i == len(l): break
            i += 1
        else:
            return 1.0
    return fuzz.token_set_ratio(a, b) / 100.0
