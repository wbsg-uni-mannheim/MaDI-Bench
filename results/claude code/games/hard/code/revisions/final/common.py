"""Shared normalization helpers (deterministic)."""
import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------- noise-aware distance
_ROWS = ["1234567890-", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
_POS = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        _POS[ch] = (r, c + (0.5 * r if r > 0 else 0))
KB = {}
for a, (ra, ca) in _POS.items():
    KB[a] = {b for b, (rb, cb) in _POS.items() if a != b and abs(ra - rb) <= 1 and abs(ca - cb) <= 1.01}
OCR = [("0", "o"), ("1", "l"), ("1", "i"), ("l", "i"), ("5", "s"), ("3", "e"), ("8", "b"), ("6", "g"),
       ("h", "b"), ("c", "e"), ("a", "o"), ("u", "v"), ("9", "g"), ("2", "z"), ("4", "a")]
CONF = set()
for a, b in OCR:
    CONF.add((a, b)); CONF.add((b, a))


def sub_cost(a, b):
    if a == b:
        return 0.0
    if (a, b) in CONF or b in KB.get(a, ()):
        return 0.35
    return 1.0


@lru_cache(maxsize=2_000_000)
def wdist(s, t):
    """Weighted Levenshtein: cheap keyboard-neighbour / OCR substitutions."""
    n, m = len(s), len(t)
    prev = [j * 1.0 for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * 1.0] + [0.0] * m
        a = s[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub_cost(a, t[j - 1]))
        prev = cur
    return prev[m]


def wsim(s, t):
    if not s or not t:
        return 0.0
    return 1.0 - wdist(s, t) / max(len(s), len(t))


# ---------------------------------------------------------------- years / numbers
YEAR_RE = re.compile(r"(19[6-9]\d|20[0-2]\d)")


def parse_year(v):
    if not v:
        return None
    m = YEAR_RE.findall(str(v))
    if not m:
        return None
    y = int(m[0])
    return y if 1960 <= y <= 2024 else None


def parse_num(v, lo, hi, integer=False):
    if v is None:
        return None
    s = str(v).strip().replace(" ", "").replace(",", ".")
    if s.lower() in ("", "tbd", "nan"):
        return None
    if not re.fullmatch(r"\d+(\.\d*)?|\.\d+", s):
        return None
    x = float(s)
    if not (lo <= x <= hi):
        return None
    return int(round(x)) if integer else round(x, 1)


# ---------------------------------------------------------------- titles
ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9",
         "x": "10", "xi": "11", "xii": "12", "xiii": "13", "xiv": "14", "xv": "15"}
DISAMBIG_RE = re.compile(r"\s*\(([^)]*)\)\s*$")


def strip_disambig(name):
    """dbpedia article titles carry '(1993 video game)' style disambiguators."""
    m = DISAMBIG_RE.search(name)
    if m and re.search(r"video game|arcade game|game|series|franchise|\b(19|20)\d\d\b", m.group(1), re.I):
        return name[: m.start()].strip()
    return name.strip()


def norm_title(t):
    t = unicodedata.normalize("NFKD", str(t))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


NUMTOK = re.compile(r"[a-z]{0,2}\d+(?:[a-z]{1,2}\d*)?")


def title_numbers(nt):
    """identity-bearing numeric tokens (arabic, roman, or short alnum like 2k3, x4, 3d).
    Tokens where digits sit inside longer words (OCR noise such as '0ddworld') are ignored."""
    out = []
    for tok in nt.split():
        if tok in ROMAN and tok != "x":
            out.append(ROMAN[tok])
        elif NUMTOK.fullmatch(tok):
            out += re.findall(r"\d+", tok)
    return tuple(out)


def numbers_compatible(a, b):
    na, nb = title_numbers(a), title_numbers(b)
    if sorted(int(x) for x in na) == sorted(int(x) for x in nb):
        return True
    return "".join(na) == "".join(nb) and len("".join(na)) > 0


# ---------------------------------------------------------------- platforms
PLAT = {
    "PlayStation": ["playstation", "ps", "ps1", "psone", "ps one", "sony playstation", "playstation 1", "psx",
                    "play station", "playstation classic"],
    "PlayStation 2": ["ps2", "playstation 2", "ps 2", "p s2", "sony playstation 2"],
    "PlayStation 3": ["ps3", "playstation 3", "ps 3", "p s3", "sony playstation 3"],
    "PlayStation 4": ["ps4", "playstation 4", "ps 4", "p s4", "playstation 4 pro"],
    "PlayStation 5": ["ps5", "playstation 5", "ps 5", "playstation5"],
    "PlayStation Portable (PSP)": ["psp", "playstation portable", "portable playstation", "ps portable", "portable"],
    "PlayStation Vita": ["ps vita", "playstation vita", "vita", "psvita", "playstation playstation vita",
                         "vita playstation", "playstationvita", "psv", "playstation tv"],
    "Xbox": ["xbox", "microsoft xbox", "xb", "original xbox"],
    "Xbox 360": ["xbox 360", "x360", "360", "xbox360", "xbox 360 console", "microsoft xbox 360", "xbox live arcade",
                 "xbla"],
    "Xbox One": ["xbox one", "xone", "xb1", "one", "xbox one s", "xbox one x", "xboxone"],
    "Xbox Series X": ["xbox series x", "xbox series x s", "xsx", "xbox series x and series s", "xbox series",
                      "series x", "xbox x", "xsx s", "xbox x s", "xbox series s", "xbox series x and s"],
    "Wii": ["wii", "nintendo wii", "wiiware", "wii ware"],
    "Wii U": ["wii u", "wiiu", "u", "nintendo wii u"],
    "Nintendo Switch": ["switch", "nintendo switch", "ns"],
    "Nintendo DS": ["ds", "nintendo ds", "nds", "double strike", "deutsche schule", "displayport", "dsi",
                    "nintendo dsi", "dsiware"],
    "Nintendo 3DS": ["3ds", "nintendo 3ds", "new 3ds", "new nintendo 3ds", "3ds nintendo"],
    "Game Boy Advance": ["gba", "game boy advance", "gameboy advance", "boy advance", "game advance",
                         "advance boy game"],
    "Game Boy": ["game boy", "gb", "gameboy"],
    "Game Boy Color": ["game boy color", "gbc", "game color", "color boy game", "gameboy color"],
    "Nintendo GameCube": ["gamecube", "nintendo gamecube", "gc", "ngc", "game cube"],
    "Nintendo 64": ["nintendo 64", "n64", "64 nintendo"],
    "Nintendo Entertainment System (NES)": ["nes", "nintendo entertainment system", "famicom", "family computer",
                                            "famicom nintendo entertainment system"],
    "Super Nintendo (SNES)": ["snes", "super nintendo", "super nintendo entertainment system", "super famicom",
                              "snintendo entertainment system", "nintendo super"],
    "Sega Dreamcast": ["dreamcast", "sega dreamcast", "dc"],
    "Sega Saturn": ["saturn", "sega saturn", "sega saturn console", "saturn sega"],
    "Sega Genesis / Mega Drive": ["sega genesis", "genesis", "mega drive", "sega mega drive", "mega drive genesis",
                                  "sega mega drive genesis", "sega megadrive", "genesis sega", "sega sega mega drive",
                                  "megadrive"],
    "Sega Master System": ["master system", "sega master system", "system master"],
    "Sega Game Gear": ["game gear", "sega game gear", "gg", "gear game"],
    "Windows PC": ["pc", "microsoft windows", "windows", "personal computer", "windows pc", "ms windows",
                   "windows microsoft", "microsoftwindows", "ibm pc", "ibm pc compatible", "ibm personal computer",
                   "windows 95", "windows xp", "windows 98", "windows 7", "windows 8", "windows 10", "win",
                   "computer personal", "steam", "steam service", "microsoft windows xp", "windows 2000"],
    "macOS": ["mac", "macos", "os x", "mac os x", "macintosh", "mac os", "classic mac os", "apple macintosh",
              "mac os x 10 5", "osx", "mac os system", "mac osx", "os mac", "os mac x", "mac x os", "x os mac"],
    "Linux / SteamOS": ["linux", "gnu linux", "steamos"],
    "iOS (iPhone)": ["ios", "iphone", "ios apple", "iphone os", "ipod touch", "apple ios", "ios by apple"],
    "iPadOS (iPad)": ["ipados", "ipad", "ipad 2"],
    "Android": ["android", "android operating system", "android os", "android system", "android operating",
                "operating android system", "system operating android", "android system operating"],
    "Google Stadia": ["stadia", "google stadia", "stadia google"],
    "Amazon Luna": ["amazon luna", "luna amazon", "amazon luna cloud gaming"],
    "Oculus Quest": ["oculus quest"],
    "Oculus Rift": ["oculus rift"],
    "PlayStation VR": ["playstation vr"],
    "Arcade Cabinet": ["arcade", "arcade game", "arcade video game", "game arcade", "arcade games", "video arcade",
                       "arcade cabinet"],
    "MS-DOS": ["ms dos", "dos", "disk operating system"],
    "Commodore 64": ["commodore 64", "c64", "64 commodore"],
    "Amiga": ["amiga", "commodore amiga", "amigaos"],
    "Atari ST": ["atari st", "st atari"],
    "ZX Spectrum": ["zx spectrum", "spectrum zx"],
    "Amstrad CPC": ["amstrad cpc", "cpc amstrad", "cpc"],
    "Ouya": ["ouya", "ouya console"],
    # digital storefronts of a single console generation -> that console
    "PlayStation 3 ": ["playstation network", "psn", "ps network"],
}
ALIAS = {}
for canon, al in list(PLAT.items()):
    canon = canon.strip()
    for a in al + [canon.lower()]:
        ALIAS[" ".join(re.sub(r"[^a-z0-9]+", " ", a.lower()).split())] = canon
ALIAS_SORTED = {" ".join(sorted(k.split())): v for k, v in ALIAS.items()}
# ambiguous/uninformative strings -> unknown
PLAT_UNKNOWN = {"", "p", "nintendo", "microsoft", "sony", "sega", "game", "console", "system", "x", "i", "3", "5",
                "2", "4", "64", "os", "unchanged", "operating system", "apple", "google", "mobile", "boy", "super",
                "d", "s", "o"}


def plat_key(v):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(v).lower()).split())


# real platforms that the noise-tolerant matcher would otherwise pull onto a nearby alias
DISTINCT = {"windows phone", "windows phone 7", "windows phone 8", "windows mobile", "apple ii", "apple iigs",
            "apple ii series", "neo geo system", "neo geo", "game com", "amstrad pcw", "commodore 128",
            "nintendo e reader", "vc", "virtual console", "wipi", "amazon kindle", "xbox live",
            "sega mega cd", "mega cd", "sega cd", "msx", "msx2", "mobile phone",
            "sega 32x", "32x", "gp32", "cd i", "playstation home", "r zone", "rzone", "amstrad", "nintendo system",
            "blackberry", "blackberry operating system", "macintosh operating systems", "atari", "commodore",
            "pc engine", "pc fx", "pc 88", "pc 98", "neo geo cd", "neo geo aes", "windows mobile phone",
            "windows phone 8", "windows ce", "turbografx 16", "3do", "n gage", "ipod game", "amazon fire tv",
            "cp m", "vhs", "os 2", "bada", "bada operating system", "ii apple", "iigs apple", "phone", "dvd", "cd"}


def _split_multi(k_raw):
    parts = [p for p in re.split(r"\s+/\s+|;|,\s+", str(k_raw)) if p.strip()]
    return parts[0] if parts else str(k_raw)


def canon_platform(v, prior=None):
    """Return (canonical, method). prior: dict canon->frequency (same source) used for tie-breaks."""
    v = _split_multi(v)
    k = plat_key(v)
    if k in PLAT_UNKNOWN:
        return None, "unknown"
    if k in ALIAS:
        return ALIAS[k], "alias"
    ks = " ".join(sorted(k.split()))
    if ks in ALIAS_SORTED:
        return ALIAS_SORTED[ks], "alias_sorted"
    kn = k.replace(" ", "")
    for a, c in ALIAS.items():
        if a.replace(" ", "") == kn:
            return c, "alias_nospace"
    if k in DISTINCT or ks in {" ".join(sorted(x.split())) for x in DISTINCT} or k.split(" ")[0] in ("windows", "apple", "neo", "mobile") and k not in ALIAS:
        return None, "distinct"

    def pick(cands, method):
        if len(cands) == 1:
            return next(iter(cands)), method
        if prior:
            ranked = sorted(cands, key=lambda c: -prior.get(c, 0))
            if prior.get(ranked[0], 0) >= 5 * max(1, prior.get(ranked[1], 0)):
                return ranked[0], method + "_prior"
        return None, "ambiguous"

    # truncation: key is a prefix of an alias
    if len(kn) >= 3:
        cands = {c for a, c in ALIAS.items() if a.replace(" ", "").startswith(kn) and len(a.replace(" ", "")) > len(kn)}
        if cands:
            return pick(cands, "prefix")
    # noise-aware fuzzy match
    best = []
    for a, c in ALIAS.items():
        a2 = a.replace(" ", "")
        if abs(len(a2) - len(kn)) > (0 if len(kn) <= 3 else max(2, len(a2) // 3)):
            continue
        best.append((wsim(kn, a2), c))
    if not best:
        return None, "unmapped"
    best.sort(key=lambda x: -x[0])
    top = best[0][0]
    L = len(kn)
    thr = 0.64 if L <= 3 else (0.72 if L <= 6 else 0.68)
    if top < thr:
        return None, "unmapped"
    return pick({c for s, c in best if s >= top - 1e-9}, "fuzzy")


# ---------------------------------------------------------------- ESRB
ESRB_ALIAS = {"e": "E", "everyone": "E", "k a": "E", "ka": "E", "t": "T", "teen": "T", "m": "M",
              "mature": "M", "mature 17": "M", "mature 17 and up": "M", "17 mature": "M", "e10": "E10+",
              "e10 plus": "E10+", "everyone 10": "E10+", "everyone 10 plus": "E10+", "10 everyone": "E10+",
              "ao": "AO", "adults only": "AO", "adults only 18": "AO", "rp": "RP", "rating pending": "RP",
              "rp lm17": "RP-LM17", "ec": None}
ESRB_TARGETS = {"E": "everyone", "T": "teen", "M": "mature 17", "E10+": "everyone 10"}


def canon_esrb(v):
    raw = str(v).strip()
    if not raw:
        return None, "empty"
    k = " ".join(re.sub(r"[^a-z0-9]+", " ", raw.lower()).split())
    if k in ESRB_ALIAS:
        return ESRB_ALIAS[k], "alias"
    if k.replace(" ", "") in ("e10", "el0", "e1o", "elo"):
        return "E10+", "alias"
    if len(raw) == 1:
        c = raw.lower()
        opts = [t for t in ("E", "T", "M") if c in KB.get(t.lower(), ()) or (c, t.lower()) in CONF]
        return (opts[0], "kb1") if len(opts) == 1 else (None, "ambiguous")
    if raw.endswith("+") and len(raw) == 4:          # noisy 'E10+'
        return "E10+", "pattern"
    if "17" in raw or "l7" in raw:
        return "M", "pattern"
    kn = k.replace(" ", "")
    sc = sorted(((wsim(kn, w.replace(" ", "")), c) for c, w in ESRB_TARGETS.items()), reverse=True)
    if sc[0][0] >= 0.6 and sc[0][0] - sc[1][0] > 0.1:
        return sc[0][1], "fuzzy"
    if len(raw) == 4 and raw[0].lower() in KB["t"] | {"t"}:
        return "T", "pattern"
    return None, "unmapped"


# ---------------------------------------------------------------- abbreviations (noise: 'GTA IV', 'THPS2')
ACR_RE = re.compile(r"^[A-Z]{2,6}\d{0,2}$")
ROMAN_UP = {k.upper() for k in ROMAN}


def acr_tokens(raw):
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(raw)) if ACR_RE.match(t) and t not in ROMAN_UP]


def _numtok(tl):
    if tl.isdigit():
        return str(int(tl))
    if tl in ROMAN and tl not in ("x", "v"):
        return ROMAN[tl]
    return None


def initials_key(ntitle):
    out = []
    for w in ntitle.split():
        n = _numtok(w)
        out.append(n if n is not None else w[0])
    return "".join(out)


def acronym_key(raw):
    """initials key of a title in which upper-case abbreviation tokens are expanded letter by letter"""
    if not acr_tokens(raw):
        return None
    out = []
    for t in re.findall(r"[A-Za-z0-9]+", str(raw)):
        if ACR_RE.match(t) and t not in ROMAN_UP:
            m = re.match(r"([A-Z]+)(\d*)", t)
            out.append(m.group(1).lower() + (str(int(m.group(2))) if m.group(2) else ""))
        else:
            tl = t.lower()
            n = _numtok(tl)
            out.append(n if n is not None else tl[0])
    return "".join(out)
