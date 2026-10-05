import re
ROMAN = {'ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10','xi':'11','xii':'12','xiii':'13','xiv':'14','xv':'15','xvi':'16'}
FILLER = {'the','a','an','game','video','videogame','dreamworks','disney','disneys','pixar','pixars','tom','clancys','sid','meiers','james','bond','marvel','and'}
def toks(s): return re.findall(r'[a-z0-9]+', s.lower().replace("'", "").replace("&", " and "))
def nums(s):
    out = set()
    for t in toks(s):
        if t in ROMAN: out.add(ROMAN[t]); continue
        for g in re.findall(r'\d+', t): out.add(g.lstrip('0') or '0')
    return out
def core(s): return frozenset(t for t in toks(s) if t not in FILLER)
