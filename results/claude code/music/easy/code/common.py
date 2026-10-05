import re, unicodedata, ast
MOJI = re.compile(r'[ÃÂâ][\x80-\xbf€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ¡-¿]')
def fix_moji(s):
    if not isinstance(s, str): return s
    for _ in range(3):
        if not MOJI.search(s): break
        ok = False
        for enc in ('cp1252', 'latin1'):
            try:
                s2 = s.encode(enc).decode('utf8'); s = s2; ok = True; break
            except Exception: pass
        if not ok:
            # partial repair: fix decodable runs only
            def rep(m):
                try: return m.group(0).encode('cp1252').decode('utf8')
                except Exception: return m.group(0)
            s2 = re.sub(r'(?:[ÃÂâ][\x80-\xbf€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ¡-¿]{1,2})+', rep, s)
            if s2 == s: break
            s = s2
    return s
TR=str.maketrans({'ø':'o','Ø':'O','æ':'ae','Æ':'AE','ß':'ss','ł':'l','Ł':'L','đ':'d','Đ':'D','œ':'oe','ı':'i'})
def strip_acc(s):
    s=s.translate(TR)
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
def key(s):
    """comparison-only folding: lowercase, strip accents/punctuation"""
    if not isinstance(s, str): return ''
    s = strip_acc(fix_moji(s)).lower().replace('&', ' and ')
    s = re.sub(r"['’`]", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return s.strip()
def parse_list(v):
    if not isinstance(v, str) or not v.strip(): return []
    try:
        x = ast.literal_eval(v)
        if isinstance(x, (list, tuple)): return [str(t) for t in x]
    except Exception: pass
    return [v]
