"""Shared normalization helpers (used by s2 and later stages)."""
import re, unicodedata

ROMAN = {'ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10',
         'xi':'11','xii':'12','xiii':'13','xiv':'14','xv':'15','xvi':'16'}

DISAMBIG = re.compile(r"\s*\(([^()]*)\)\s*$")

def strip_disambig(title):
    """Wikipedia-style trailing '(1988 video game)' -> (title, year|None, tag|None)."""
    t = title.strip(); year = None; tag = None
    m = DISAMBIG.search(t)
    if m:
        inner = m.group(1)
        if re.search(r"(video ?game|game|series|franchise|arcade|software|computer|console|film|\d{4})", inner, re.I):
            tag = inner
            y = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", inner)
            if y: year = int(y.group(1))
            t = t[:m.start()].strip()
    return t, year, tag

def fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()

def title_tokens(title):
    s = fold(title)
    s = s.replace('&', ' and ').replace("'", '').replace('’','')
    s = re.sub(r"[^a-z0-9]+", ' ', s)
    toks = [ROMAN.get(t, t) for t in s.split()]
    toks = ['vs' if t == 'versus' else t for t in toks]
    return toks

def title_key(title):
    toks = [t for t in title_tokens(title) if t not in ('the','a','an')]
    return ' '.join(toks)

# ---------------- platforms ----------------
DISPLAY = {'pc':'PC','ps1':'PlayStation','ps2':'PS2','ps3':'PS3','ps4':'PS4','ps5':'PlayStation 5',
 'psp':'Playstation Portable','vita':'PlayStation Vita','xbox':'Xbox','x360':'Xbox 360','xone':'Xbox One',
 'xsx':'Xbox Series X','wii':'Wii','wiiu':'Wii U','switch':'Switch','gc':'GameCube','n64':'Nintendo 64',
 'ds':'DS','3ds':'3DS','gba':'Game Boy Advance','dc':'Dreamcast','stadia':'Stadia'}

_RULES = [
 (r"^(microsoft )?windows( pc| \d.*| xp.*| vista| me| 9x| nt| 2000| 7| 8(\.1)?| 10| 95| 98| 3\.\w+| store)?$|^pc$|^personal computer$|^microsoft windows.*$|^games for windows.*", 'pc'),
 (r"^(sony )?playstation( \(console\))?$", 'ps1'),
 (r"^(ps2|playstation 2)$", 'ps2'), (r"^(ps3|playstation 3)$", 'ps3'), (r"^(ps4|playstation 4)$", 'ps4'),
 (r"^(ps5|playstation 5)$", 'ps5'),
 (r"^(playstation portable|playstation potable|psp)$", 'psp'),
 (r"^(playstation vita\w*|ps ?vita|psv)$", 'vita'),
 (r"^xbox$", 'xbox'), (r"^(xbox 360( \(console\))?|x360)$", 'x360'), (r"^(xbox one|xone)$", 'xone'),
 (r"^xbox series.*$", 'xsx'),
 (r"^wii$", 'wii'), (r"^(wii ?u)$", 'wiiu'), (r"^(nintendo )?switch$", 'switch'),
 (r"^(nintendo )?gamecube$|^gc$", 'gc'), (r"^nintendo 64$|^n64$", 'n64'),
 (r"^(nintendo )?ds$", 'ds'), (r"^(new )?(nintendo )?(new )?3ds( family)?$", '3ds'),
 (r"^game ?boy advance$|^gba$", 'gba'), (r"^game boy color$", 'gbc'), (r"^game boy$", 'gb'),
 (r"^(sega )?dreamcast$|^dc$", 'dc'), (r"^(google )?stadia$", 'stadia'),
 (r"^(super nintendo.*|snintendo entertainment system|super famicom|snes)$", 'snes'),
 (r"^(nintendo entertainment system|family computer|famicom|famicom/nintendo entertainment system|nes)$", 'nes'),
 (r"^(sega )?(genesis|mega ?drive|megadrive)(/.*)?$|^sega mega drive/.*$|^mega drive/genesis$", 'genesis'),
 (r"^(sega )?master system$|^sega mark iii$", 'sms'),
 (r"^(sega )?game gear$", 'gamegear'), (r"^(sega )?saturn$", 'saturn'),
 (r"^(arcade|video arcade|amusement arcade|arcade (video )?games?|arcade cabinet)$", 'arcade'),
 (r"^(apple )?ios( \(apple\)| \d+)?$|^(apple )?iphone( os.*)?$|^ipad( \d)?$|^ipados$", 'ios'),
 (r"^android.*$", 'android'),
 (r"^(mac ?os.*|os ?x|os/x|macintosh|apple macintosh|apple mac|classic mac os|macintosh operating systems)$", 'mac'),
 (r"^(linux|steamos)$", 'linux'), (r"^(ms-dos|dos|ibm pc dos)$", 'dos'),
]
_RULES = [(re.compile(p), k) for p, k in _RULES]

def platform_key(raw):
    if raw is None or (isinstance(raw, float) and raw != raw): return None
    s = fold(str(raw)).strip()
    if not s: return None
    for p, k in _RULES:
        if p.match(s): return k
    return 'other:' + re.sub(r"\s+", ' ', s)

SALES_FIX = {'XOne':'Xbox One','WiiU':'Wii U','DC':'Dreamcast'}

ESRB_OK = {'E','E10+','T','M','AO','RP','RP-LM17'}
def esrb(v):
    if v is None: return None
    v = str(v).strip().upper()
    if v == 'K-A': return 'E'
    return v if v in ESRB_OK else None
