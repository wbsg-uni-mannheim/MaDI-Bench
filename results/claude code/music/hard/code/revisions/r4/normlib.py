"""Shared normalization helpers (deterministic, no external data)."""
import re, ast, unicodedata
from collections import Counter
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------- text basics
MOJI_MARKERS = ("Ã", "Â", "â€", "Ä", "Å", "Ð", "Ñ")

def fix_mojibake(s):
    if not isinstance(s, str) or not any(m in s for m in MOJI_MARKERS):
        return s
    cur = s
    for _ in range(3):
        try:
            nxt = cur.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                nxt = cur.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
        if nxt == cur:
            break
        cur = nxt
        if not any(m in cur for m in MOJI_MARKERS):
            break
    return cur

def is_moji(s):
    return isinstance(s, str) and any(m in s for m in MOJI_MARKERS)

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def key(s):
    """comparison key: lowercase ascii alnum tokens"""
    if not isinstance(s, str):
        return ""
    s = strip_accents(fix_mojibake(s)).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def clean_ws(s):
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- titles
# promo/noise tags observed only in single sources (lastfm prefixes, musicbrainz suffixes)
TITLE_NOISE_PATTERNS = [
    r"^\*popular\*\s*", r"^\[explicit\]\s*", r"^-\s*new\s*-\s*", r"^\(release\)\s*", r"^\(album\)\s*", r"^\(orig\.\)\s*", r"^Album:\s+",
    r"\s*\((?:orig\.|release|album)\)\s*$",
]

def clean_title(t, artist=None):
    """Return display title with injected promo tags / 'Artist -  ' prefixes removed."""
    if not isinstance(t, str) or not t.strip():
        return None
    t = fix_mojibake(t)
    changed = True
    while changed:
        changed = False
        for p in TITLE_NOISE_PATTERNS:
            n = re.sub(p, "", t, flags=re.I)
            if n != t:
                t, changed = n, True
    # lastfm 'Artist -  Title' (double space after dash is the tell)
    m = re.match(r"^(.{1,80}?) -  (.+)$", t)
    if m:
        t = m.group(2)
    t = clean_ws(t)
    return t or None

def title_prefix_artist(t):
    if not isinstance(t, str):
        return None
    t = fix_mojibake(t)
    for p in TITLE_NOISE_PATTERNS[:7]:
        t = re.sub(p, "", t, flags=re.I)
    m = re.match(r"^(.{1,80}?) -  (.+)$", t)
    return clean_ws(m.group(1)) if m else None

# ---------------------------------------------------------------- artists
def clean_artist(a, src):
    """display-level cleanup; returns list of artist names (discogs uses '|')."""
    if not isinstance(a, str) or not a.strip():
        return []
    a = fix_mojibake(a)
    parts = a.split("|") if src == "discogs" else [a]
    out = []
    for p in parts:
        p = re.sub(r"\s*\(\d+\)\s*$", "", p)            # discogs disambiguator
        p = re.sub(r"\s*\(and others\)\s*$", "", p, flags=re.I)
        p = re.sub(r"\s+feat\.?\s*$", "", p, flags=re.I)
        p = re.sub(r"^feat\.?\s+", "", p, flags=re.I)
        p = clean_ws(p)
        m = re.match(r"^([^,]+),\s*([^,]+)$", p)
        if m and src == "musicbrainz":                 # 'Garnier, Laurent' / 'Charlatans, The'
            p = f"{m.group(2)} {m.group(1)}"
        if p:
            out.append(p)
    return out

# ---------------------------------------------------------------- dates
OCR_DIGIT = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1", "i": "1", "r": "1", "s": "5", "S": "5",
                           "B": "8", "g": "9", "z": "2", "Z": "2"})

def parse_date(s):
    """-> (iso 'YYYY-MM-DD' or None, year or None, precision 'day'|'month'|'year'|None)"""
    if not isinstance(s, str) or not s.strip():
        return None, None, None
    s = s.strip().translate(OCR_DIGIT)
    def ok(y, mo, d):
        return 1900 <= y <= 2025 and 1 <= mo <= 12 and 1 <= d <= 31
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return (f"{y:04d}-{mo:02d}-{d:02d}", y, "day") if ok(y, mo, d) else (None, None, None)
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        return (f"{y:04d}-{mo:02d}-{d:02d}", y, "day") if ok(y, mo, d) else (None, None, None)
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        mo, d, y = map(int, m.groups())
        return (f"{y:04d}-{mo:02d}-{d:02d}", y, "day") if ok(y, mo, d) else (None, None, None)
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        y, mo = map(int, m.groups())
        return (f"{y:04d}-{mo:02d}-01", y, "month") if ok(y, mo, 1) else (None, None, None)
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(s)
        return (f"{y:04d}-01-01", y, "year") if ok(y, 1, 1) else (None, None, None)
    return None, None, None

# ---------------------------------------------------------------- durations
def parse_duration(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().lower()
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?", s)
    if m and any(m.groups()):
        h, mi, se = (int(x) if x else 0 for x in m.groups())
        v = h * 3600 + mi * 60 + se
    elif re.fullmatch(r"\d+:\d{1,2}:\d{2}", s):
        h, mi, se = map(int, s.split(":"))
        v = h * 3600 + mi * 60 + se
    elif re.fullmatch(r"\d+:\d{2}", s):
        mi, se = map(int, s.split(":"))
        v = mi * 60 + se
    elif re.fullmatch(r"\d{1,3}(?:[ ,]\d{3})+|\d+", s):
        v = int(re.sub(r"[ ,]", "", s))
    else:
        return None
    return v if v > 0 else None     # zero durations are placeholders, treated as unknown

# ---------------------------------------------------------------- tracks
def parse_tracks(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [clean_ws(fix_mojibake(str(x))) for x in v if str(x).strip()]
    except Exception:
        pass
    return []

# ---------------------------------------------------------------- countries
CANON_COUNTRY = {
    "United States of America": ["usa", "us", "united states", "united states of america", "america", "usa of america",
                                 "u s a", "u s"],
    "United Kingdom of Great Britain and Northern Ireland": ["uk", "u k", "united kingdom", "great britain", "gb", "england",
        "united kingdom of great britain and northern ireland", "uk of great britain and northern ireland",
        "united kingdom of uk and northern ireland", "britain"],
    "Germany": ["germany", "federal republic of germany", "german", "deutschland", "brd"],
    "France": ["france", "french republic"], "Japan": ["japan", "nippon"],
    "Netherlands": ["netherlands", "holland", "the netherlands", "kingdom of the netherlands", "nederland"],
    "Europe": ["europe", "europa", "eu"], "Italy": ["italy", "italia", "italian republic"],
    "Australia": ["australia", "commonwealth of australia"], "Russian Federation": ["russia", "russian federation"],
    "Spain": ["spain", "espana", "kingdom of spain"], "Sweden": ["sweden", "sverige", "kingdom of sweden"],
    "Poland": ["poland", "polska", "republic of poland"], "Belgium": ["belgium", "kingdom of belgium", "belgique"],
    "Canada": ["canada"], "Jamaica": ["jamaica"], "Brazil": ["brazil", "brasil", "federative republic of brazil"],
    "Greece": ["greece", "hellas", "hellenic republic"], "Finland": ["finland", "suomi", "republic of finland"],
    "Portugal": ["portugal", "portuguese republic"], "Switzerland": ["switzerland", "swiss confederation", "schweiz"],
    "Mexico": ["mexico", "united mexican states"], "Norway": ["norway", "norge", "kingdom of norway"],
    "Denmark": ["denmark", "danmark", "kingdom of denmark"], "New Zealand": ["new zealand", "aotearoa"],
    "Austria": ["austria", "republic of austria", "osterreich"], "Argentina": ["argentina", "republica argentina", "argentine republic"],
    "Ukraine": ["ukraine"], "South Africa": ["south africa", "republic of south africa"],
    "Indonesia": ["indonesia", "republik indonesia", "republic of indonesia"], "Ireland": ["ireland", "eire", "republic of ireland"],
    "Hungary": ["hungary", "magyarorszag"], "Malaysia": ["malaysia"], "Colombia": ["colombia", "republica de colombia", "republic of colombia"],
    "Chile": ["chile", "republic of chile"], "Bulgaria": ["bulgaria", "republic of bulgaria"],
    "Korea, Republic of": ["south korea", "korea republic of", "republic of korea", "korea"], "Yugoslavia": ["yugoslavia", "sfr yugoslavia"],
    "Turkey": ["turkey", "republic of turkey", "turkiye"], "Taiwan": ["taiwan"], "Scandinavia": ["scandinavia"],
    "Venezuela": ["venezuela"], "Israel": ["israel", "state of israel"], "Thailand": ["thailand", "kingdom of thailand"],
    "Philippines": ["philippines", "republic of the philippines"], "Czech Republic": ["czech republic", "czechia"],
    "Singapore": ["singapore", "republic of singapore"], "Romania": ["romania"], "India": ["india", "republic of india"],
    "China": ["china", "people s republic of china", "prc"], "Hong Kong": ["hong kong"], "Australasia": ["australasia"],
    "Czechoslovakia": ["czechoslovakia"], "Slovenia": ["slovenia"], "Peru": ["peru"], "Lithuania": ["lithuania"],
    "Croatia": ["croatia", "republika hrvatska"], "Benelux": ["benelux"], "Uruguay": ["uruguay"], "Belarus": ["belarus"], "Estonia": ["estonia"],
    "German Democratic Republic (GDR)": ["german democratic republic gdr", "gdr", "east germany", "german democratic republic"],
    "Luxembourg": ["luxembourg"], "Nigeria": ["nigeria"], "Slovakia": ["slovakia"], "Saudi Arabia": ["saudi arabia"],
    "United Arab Emirates": ["united arab emirates", "uae"], "Iceland": ["iceland"], "Serbia": ["serbia"], "Lebanon": ["lebanon"],
    "USSR": ["ussr", "soviet union"], "Asia": ["asia"], "South East Asia": ["south east asia"], "Africa": ["africa"],
    "South America": ["south america"], "Barbados": ["barbados"], "Latvia": ["latvia"], "Bosnia & Herzegovina": ["bosnia and herzegovina"],
    "Republic of China": ["republic of china"], "Worldwide": ["worldwide"], "Egypt": ["egypt"], "Cuba": ["cuba"],
    "Trinidad & Tobago": ["trinidad and tobago"], "Ecuador": ["ecuador"], "Puerto Rico": ["puerto rico"], "Kenya": ["kenya"],
    "Macedonia": ["macedonia"], "Iran": ["iran"], "Pakistan": ["pakistan"], "Vietnam": ["vietnam"], "Costa Rica": ["costa rica"],
    "Ghana": ["ghana"], "Kenya": ["kenya", "republic of kenya"], "Tunisia": ["tunisia"], 
    "Moldova, Republic of": ["moldova", "moldova republic of"], "Reunion": ["reunion"], "Cyprus": ["cyprus"], "Malta": ["malta"], "Guatemala": ["guatemala"], "Panama": ["panama"],
    "North America": ["north america"], "Middle East": ["middle east"], "Zimbabwe": ["zimbabwe"], "Bolivia": ["bolivia"],
}
ALIAS2C = {}
for c, al in CANON_COUNTRY.items():
    for a in al + [key(c)]:
        ALIAS2C[a] = c
SORTED_ALIAS = {}
for a, c in ALIAS2C.items():
    SORTED_ALIAS.setdefault(" ".join(sorted(a.split())), c)
OCR_LET = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g", "8": "b", "3": "e", "4": "a", "7": "t"})
_ALIAS_LIST = list(ALIAS2C)
def skel(s):
    return s.replace("e", "c").replace("a", "o")
SKEL_ALIAS = {}
for a, c in ALIAS2C.items():
    if len(a) >= 4:
        SKEL_ALIAS.setdefault(skel(a), c)

def norm_country(s):
    """-> (canonical or None, method). Multi-region values ('UK & Europe') are kept as their own label."""
    if not isinstance(s, str) or not s.strip():
        return None, None
    raw = fix_mojibake(s).strip()
    k = key(raw)
    if k in ALIAS2C:
        return ALIAS2C[k], "alias"
    ks = " ".join(sorted(k.split()))
    if ks in SORTED_ALIAS:
        return SORTED_ALIAS[ks], "permutation"
    if re.search(r"&|,| and ", raw) and not re.search(r"(?i)kingdom|republic|bosnia|trinidad", raw):
        parts = [p for p in re.split(r"\s*(?:&|,|\band\b)\s*", raw) if p.strip()]
        canon = [norm_country(p)[0] for p in parts]
        if len(parts) > 1 and all(canon):
            return raw.strip(), "multi-region"
    k2 = key(raw.translate(OCR_LET))
    if k2 in ALIAS2C:
        return ALIAS2C[k2], "ocr"
    if len(k2) < 4:
        return None, "unresolved"
    # OCR e/c and a/o confusions: compare on a skeleton where e->c, a->o
    sk = skel(k2)
    if sk in SKEL_ALIAS:
        return SKEL_ALIAS[sk], "ocr-skeleton"
    # fuzzy against aliases (handles OCR e/c a/o swaps, truncations, duplicated words)
    best = process.extractOne(k2, _ALIAS_LIST, scorer=fuzz.ratio)
    if best and best[1] >= 80 and len(best[0]) >= 4:
        return ALIAS2C[best[0]], "fuzzy"
    best = process.extractOne(k2, _ALIAS_LIST, scorer=fuzz.token_set_ratio)
    if best and best[1] >= 90 and len(best[0]) >= 8 and len(k2) >= 0.6 * len(best[0]):
        return ALIAS2C[best[0]], "fuzzy-tokens"
    # truncated prefix of a long alias ('United States of Amer', 'Netherlan', 'Germa')
    cands = {ALIAS2C[a] for a in _ALIAS_LIST if a.startswith(k2) and len(k2) >= 4}
    if len(cands) == 1:
        return cands.pop(), "prefix"
    return None, "unresolved"

# ---------------------------------------------------------------- genres (discogs vocabulary)
GENRE_ATOMS = ["Electronic", "Rock", "Pop", "Funk / Soul", "Folk, World, & Country", "Jazz", "Hip Hop", "Classical",
               "Reggae", "Blues", "Stage & Screen", "Latin", "Non-Music", "Children's", "Brass & Military"]
GENRE_KEYWORDS = [  # (regex over lowercased OCR-fixed text, atom)
    (r"non[\s-]*mus", "Non-Music"), (r"electr|electro", "Electronic"), (r"rock\s*(?:&|and)\s*roll", "Rock"),
    (r"\brock|\broc\b|\bro\s?ck", "Rock"), (r"\bpop\b|\bpop\s*music|\bpo\b", "Pop"), (r"funk|soul|r&b|r and b", "Funk / Soul"),
    (r"folk|world|country", "Folk, World, & Country"), (r"jazz", "Jazz"), (r"hip|hop|rap\b", "Hip Hop"),
    (r"classic", "Classical"), (r"reggae", "Reggae"), (r"blues", "Blues"), (r"stage|screen|theat|film|soundtrack", "Stage & Screen"),
    (r"latin", "Latin"), (r"child|\bkids\b", "Children's"), (r"\bmetal\b|\bpunk\b", "Rock"), (r"caribbean", "Reggae"), (r"brass|military", "Brass & Military"),
]
GENRE_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "8": "b"})

def _atom_fuzzy(tok):
    t = tok.lower().translate(GENRE_OCR).replace(" ", "").replace("|", "")
    if len(t) < 2:
        return None
    cands = {a: fuzz.ratio(t, re.sub(r"[^a-z]", "", a.lower())) for a in GENRE_ATOMS}
    a, sc = max(cands.items(), key=lambda x: x[1])
    if sc >= 70:
        return a
    pref = [a for a in GENRE_ATOMS if re.sub(r"[^a-z]", "", a.lower()).startswith(re.sub(r"[^a-z]", "", t))]
    if len(pref) == 1 and len(t) >= 2:
        return pref[0]
    return None

def genre_atoms(raw):
    """-> list of discogs genre atoms found in a (noisy) genre string, in order of appearance."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    found = []
    for piece in raw.split("|"):
        p = piece.strip()
        if p in GENRE_ATOMS:
            found.append((0, p)); continue
        low = p.lower().translate(GENRE_OCR)
        hits = []
        for rx, atom in GENRE_KEYWORDS:
            for m in re.finditer(rx, low):
                hits.append((m.start(), atom))
        if not hits:
            # OCR-noise / truncation: fuzzy per whitespace/slash token, then whole piece
            a = _atom_fuzzy(p)
            if a:
                hits.append((0, a))
            else:
                for tok in re.split(r"[\s/,&-]+", p):
                    a = _atom_fuzzy(tok) if len(tok) >= 3 else None
                    if a:
                        hits.append((low.find(tok.lower()), a))
        hits.sort()
        for _, a in hits:
            found.append((0, a))
    out = []
    for _, a in found:
        if a not in out:
            out.append(a)
    return out

# ---------------------------------------------------------------- vocabulary-based repair (observed values only)
INITIAL_RX = re.compile(r"^[A-Za-z]\.$")

def build_artist_vocab(names):
    """names: iterable of full artist display strings (no initials). -> index for expand_initials"""
    idx = {}
    for nm in set(names):
        toks = nm.split()
        if not toks or any(INITIAL_RX.match(t) for t in toks):
            continue
        for i, t in enumerate(toks):
            idx.setdefault((len(toks), i, t.lower()), set()).add(nm)
    return idx

def expand_initials(name, idx):
    """'T. Monochrome Set' -> 'The Monochrome Set' when exactly one observed artist name fits the pattern."""
    if not isinstance(name, str):
        return None
    toks = name.split()
    full = [(i, t) for i, t in enumerate(toks) if not INITIAL_RX.match(t)]
    if len(full) == len(toks) or not full:
        return None
    i0, t0 = full[0]
    cands = idx.get((len(toks), i0, t0.lower()), set())
    ok = []
    for c in cands:
        ct = c.split()
        if all((ct[i].lower().startswith(t[0].lower()) and len(ct[i]) > 1) if INITIAL_RX.match(t) else ct[i].lower() == t.lower()
               for i, t in enumerate(toks)):
            ok.append(c)
    ok = sorted(set(ok))
    return ok[0] if len(ok) == 1 else None

LABEL_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "8": "b", "4": "a", "7": "t"})

def build_label_fixer(counts):
    """counts: Counter of label strings. A rare label is mapped to a >=3x more frequent label whose OCR-normalised
    form is within fuzz.ratio >= 88 (spelling-noise repair). Returns dict raw -> corrected."""
    labs = [l for l, c in counts.most_common() if c >= 3]
    norm = {l: re.sub(r"\s+", "", l.lower().translate(LABEL_OCR)) for l in counts}
    fixes = {}
    frequent_norm = [norm[l] for l in labs]
    for l, c in counts.items():
        if c >= 3 and l in labs[:200]:
            continue
        best = process.extractOne(norm[l], frequent_norm, scorer=fuzz.ratio, score_cutoff=88)
        if best:
            tgt = labs[best[2]]
            strip_n = lambda x: re.sub(r"\s*\(\d+\)$", "", x)
            if (tgt != l and counts[tgt] >= 3 * c and strip_n(l) != strip_n(tgt)
                    and tgt.lower() not in l.lower() and len(tgt) >= 0.8 * len(l)):
                fixes[l] = tgt
    return fixes
