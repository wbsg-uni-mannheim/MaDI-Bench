"""Platform canonicalisation: raw source string -> canonical platform key (or None if unresolved)."""
import re
from rapidfuzz.distance import Levenshtein
ALIASES = {
 'pc': ['PC','Microsoft Windows','Windows','Personal Computer','Windows PC','Windows 95','Windows 98','Windows XP',
        'Windows 2000','Windows 7','Windows 8','Windows 8.1','Windows 10','Windows Vista','Windows 3.1','Windows 3.1x','Windows Me',
        'Windows 9x','Microsoft Windows XP','Microsoft Windows 95','Microsoft Windows 98','Microsoft Windows 2000','Games for Windows Live',
        'IBM PC compatible','Steam','Steam (service)','Desktop Computer','Windows Store','Microsoft Windows NT'],
 'dos': ['MS-DOS','DOS','Microsoft Disk Operating System','IBM PC','IBM Personal Computer','PC/AT'],
 'mac': ['MacOS','Mac OS X','OS X','Mac OS','Macintosh','Classic Mac OS','Mac','Apple Macintosh','OSX','Mac OS 9','Mac OS 8','Mac OS 7'],
 'linux': ['Linux','GNU/Linux','SteamOS','Linux / SteamOS','Linux (Steam)'],
 'ps1': ['PlayStation','PS','PSX','PS1','Sony PlayStation'],
 'ps2': ['PS2','PlayStation 2'],
 'ps3': ['PS3','PlayStation 3'],
 'ps4': ['PS4','PlayStation 4'],
 'ps5': ['PS5','PlayStation 5'],
 'psp': ['PlayStation Portable','PSP','PS Portable','Portable PlayStation'],
 'vita': ['PlayStation Vita','PS Vita','Vita','PlayStation PlayStation Vita','PlayStation TV'],
 'psn': ['PlayStation Network','PSN','PS Network','PlayStation Store'],
 'psvr': ['PlayStation VR'], 'psvr2': ['PlayStation VR2','PS VR2'],
 'xbox': ['Xbox','Microsoft Xbox','XB'],
 'x360': ['Xbox 360','X360','360','Xbox 360 S','Xbox 360 Console','Microsoft Xbox 360','Xbox 360 (console)'],
 'xone': ['Xbox One','XOne','XB1','One','Xbox One S','Xbox One X'],
 'xsx': ['Xbox Series X','Xbox Series X/S','XSX','Xbox Series X and Series S','Xbox Series X|S','XSX/S','Xbox Series X and S','Xbox Series','Xbox Series S','Xbox X','Series X'],
 'xbla': ['Xbox Live Arcade','XBLA','Xbox Live'],
 'switch': ['Switch','Nintendo Switch','Nintendo Switch Console','Nintendo Switch Lite'],
 'wii': ['Wii'], 'wiiu': ['Wii U','WiiU'],
 'wiiware': ['WiiWare'], 'vc': ['Virtual Console','Wii Virtual Console'],
 'ds': ['DS','Nintendo DS'], 'dsi': ['DSi','DSiWare','Nintendo DSi'],
 '3ds': ['3DS','Nintendo 3DS','New Nintendo 3DS','New 3DS','Nintendo 2DS','Nintendo 3DS XL'],
 'gba': ['Game Boy Advance','GBA','GameBoy Advance','Game Advance','Boy Advance'], 'gb': ['Game Boy'], 'gbc': ['Game Boy Color','GBC','GameBoy Color'],
 'gc': ['GameCube','Nintendo GameCube','GC','NGC'], 'n64': ['Nintendo 64','N64'],
 'nes': ['Nintendo Entertainment System','NES','Famicom','Family Computer','Famicom/Nintendo Entertainment System'],
 'fds': ['Famicom Disk System','Family Computer Disk System'],
 'snes': ['Super Nintendo','SNES','Super Famicom','Super NES','Super Nintendo Entertainment System','SNintendo Entertainment System'],
 'genesis': ['Sega Genesis','Mega Drive','Sega Mega Drive','Genesis','Mega Drive/Genesis','Sega Mega Drive/Genesis','Sega Genesis/Mega Drive','Genesis/Mega Drive','Sega Megadrive','GEN'],
 'saturn': ['Sega Saturn','Saturn'], 'dc': ['Dreamcast','Sega Dreamcast','DC'],
 'segacd': ['Sega CD','Sega Mega-CD','Mega CD','Mega-CD'], 'sms': ['Master System','Sega Master System'],
 'gg': ['Game Gear','Sega Game Gear'], '32x': ['32X','Sega 32X'],
 'ios': ['IOS','iOS','IOS (Apple)','IPhone','IPad','IPadOS','IPhone OS','IPod Touch','Apple iOS','IPad 2'],
 'android': ['Android (operating system)','Android','Android OS','Android (OS)','Android Operating System'],
 'mobile': ['Mobile','Phone','Mobile games','Mobile gaming','Mobile phone','Mobile game','Mobile phones','Mobile Phones','Cellphone','Mobile device'],
 'arcade': ['Arcade game','Arcade','Arcade video game','Arcade Cabinet','Arcade games'],
 'stadia': ['Google Stadia','Stadia'], 'luna': ['Amazon Luna'], 'ngage': ['N-Gage (device)','N-Gage','N-Gage (service)'],
 'atari2600': ['Atari 2600','2600'], 'jaguar': ['Atari Jaguar','Jaguar'], 'lynx': ['Atari Lynx','Lynx'],
 'ouya': ['Ouya'], 'wp': ['Windows Phone','Windows Phone 7','Windows Phone 8'],
 'tg16': ['TurboGrafx-16','PC Engine','PC Engine/TurboGrafx-16'],
}
def compact(s): return re.sub(r'[^a-z0-9]', '', s.lower())
def toksort(s): return ' '.join(sorted(re.findall(r'[a-z0-9]+', s.lower())))
CONF = str.maketrans({'5':'s','0':'o','1':'l','8':'b','6':'g','3':'e','9':'g','4':'a'})
def conf(s): return compact(s).translate(CONF)
EXACT, TOKS, CONFK = {}, {}, {}
for k, al in ALIASES.items():
    for a in al:
        EXACT[compact(a)] = k; TOKS[toksort(a)] = k; CONFK.setdefault(conf(a), set()).add(k)
ALLC = list(EXACT.items())
_cache = {}
NOFUZZ = {'wipi','windowsce','phone'}

def canon_platform(raw):
    """Returns (key, method). key None when the string cannot be resolved unambiguously."""
    if raw is None or str(raw).strip() == '': return None, 'missing'
    raw = str(raw)
    if raw in _cache: return _cache[raw]
    c = compact(raw); r = None
    if c in EXACT: r = (EXACT[c], 'exact')
    elif toksort(raw) in TOKS: r = (TOKS[toksort(raw)], 'tokens')
    else:
        # 'Nintendo Nintendo Switch', 'Switch Nintendo' style duplications
        ts = ' '.join(sorted(set(re.findall(r'[a-z0-9]+', raw.lower()))))
        if ts in TOKS: r = (TOKS[ts], 'tokens')
    if r is None:
        ks = CONFK.get(conf(raw))
        if ks and len(ks) == 1: r = (next(iter(ks)), 'ocr')
    if r is None and len(c) >= 5 and c not in NOFUZZ:
        best = {}
        for a, k in ALLC:
            if abs(len(a) - len(c)) > 1: continue
            d = Levenshtein.distance(a, c, score_cutoff=1)
            if d <= 1: best.setdefault(d, set()).add(k)
        for d in sorted(best):
            if len(best[d]) == 1: r = (next(iter(best[d])), f'edit{d}')
            break
    if r is None: r = (None, 'unresolved')
    _cache[raw] = r
    return r
