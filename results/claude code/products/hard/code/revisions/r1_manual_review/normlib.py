"""Shared normalization helpers (deterministic)."""
import re, difflib
BRANDS = {  # canonical key -> aliases (lowercase, alnum only)
 'seagate':['seagate','seagatetechnology','ironwolf','barracuda','firecuda','exos','skyhawk','hikvisionseagate','seagatelacie','maxtorseagate'],
 'wd':['wd','westerndigital','digitalwestern','wdblack','western','wdc','westerndigitalcorporation'],
 'samsung':['samsung','samsungelectronics','samsunggroup'],
 'gigabyte':['gigabyte','gb','aorus','gigabytetechnology'],
 'asus':['asus','asustek','rogstrix','asustekcomputer'],
 'kingston':['kingston','kingstontechnology','kingstontechnol','technologykingston','kingstondigital','hyperx'],
 'msi':['msi','msicomputer','msicomputercorporation'],
 'hp':['hp','hpe','hewlettpackardenterprise','hewlettpackard','hpenterprise','hewlettpackardhp'],
 'toshiba':['toshiba','toshibacorporation'],
 'zotac':['zotac','zotacinternational','zotacgmbh','zotactechnology','zotacgaming'],
 'adata':['adata','xpg','adataxpg','adatatechnology'],
 'intel':['intel','intelcorporation','intc'],
 'sandisk':['sandisk','sandiskcorporation'],
 'crucial':['crucial'], 'micron':['micron','microntechnology'],
 'pny':['pny'], 'sapphire':['sapphire','sapphiretechnology'], 'evga':['evga','evgacorporation'],
 'corsair':['corsair'], 'patriot':['patriot','patriotmemory'], 'palit':['palit','palitmicrosystems'],
 'amd':['amd','advancedmicrodevices'], 'xfx':['xfx'], 'lacie':['lacie','laciesarl'],
 'siliconpower':['siliconpower'], 'transcend':['transcend','transcendinformation'],
 'dell':['dell','dellinc','dellequallogic'], 'intenso':['intenso'], 'powercolor':['powercolor','powercolortechnology'],
 'lenovo':['lenovo','lenovogroup'], 'inno3d':['inno3d'], 'nvidia':['nvidia'], 'sabrent':['sabrent'],
 'lexar':['lexar'], 'teamgroup':['teamgroup','team'], 'asrock':['asrock'], 'gtechnology':['gtechnology'],
 'fujitsu':['fujitsu'], 'istorage':['istorage','istoragecorp'], 'maxtor':['maxtor'], 'verbatim':['verbatim'],
 'hitachi':['hitachi','hgst'], 'ibm':['ibm'], 'netapp':['netapp'], 'pioneer':['pioneer'], 'biostar':['biostar'],
 'gainward':['gainward'], 'kfa2':['kfa2','galax'], 'emtec':['emtec'], 'philips':['philips'], 'toshibamemory':[],
 'aurora':['aurora'], 'auraloop':['auraloop'], 'auroralink':['auroralink'], 'aurelia':['aurelia'], 'auralis':['auralis'],
 'neru':['neru'], 'novatek':['novatek'], 'asterloop':['asterloop'], 'wansenda':['wansenda'], 'pqi':['pqi'],
 'hikvision':['hikvision'], 'lenovothinkpad':[], 'apple':['apple'], 'kioxia':['kioxia'], 'sony':['sony'],
}
ALIAS = {al:k for k,v in BRANDS.items() for al in v}
# title-detectable brand words (word boundary match on lowercase title)
TITLE_BRAND_PAT = [(k, re.compile(r'(?<![a-z0-9])'+p+r'(?![a-z0-9])')) for k,p in [
 ('wd',r'western\s*digital'),('wd',r'wd'),('wd',r'wd_black'),('seagate','seagate'),('samsung','samsung'),('gigabyte',r'(?<!\d )gigabyte'),('gigabyte','aorus'),
 ('asus','asus'),('kingston','kingston'),('kingston','hyperx'),('msi','msi'),('hp','hpe'),('hp','hp'),('hp',r'hewlett[\s-]*packard'),('toshiba','toshiba'),
 ('zotac','zotac'),('adata','adata'),('adata','a-data'),('adata','xpg'),('intel','intel'),('sandisk','sandisk'),('crucial','crucial'),('micron','micron'),
 ('pny','pny'),('sapphire','sapphire'),('evga','evga'),('corsair','corsair'),('patriot','patriot'),('palit','palit'),('xfx','xfx'),
 ('lacie','lacie'),('siliconpower',r'silicon[\s-]*power'),('transcend','transcend'),('dell','dell'),('intenso','intenso'),('powercolor','powercolor'),
 ('lenovo','lenovo'),('inno3d','inno3d'),('sabrent','sabrent'),('lexar','lexar'),('teamgroup',r'team\s*group'),('asrock','asrock'),
 ('gtechnology',r'g-technology'),('fujitsu','fujitsu'),('istorage','istorage'),('maxtor','maxtor'),('verbatim','verbatim'),('hitachi','hitachi'),('hitachi','hgst'),
 ('gainward','gainward'),('kfa2','kfa2'),('kfa2','galax'),('ibm','ibm'),('netapp','netapp'),('biostar','biostar'),
 ('aurora','aurora'),('auraloop','auraloop'),('auroralink','auroralink'),('aurelia','aurelia'),('auralis','auralis'),('neru','neru'),('asterloop','asterloop'),
 ('seagate','ironwolf'),('seagate','barracuda'),('seagate','firecuda'),('seagate',r'exos'),('seagate','skyhawk'),
 ('nvidia','nvidia'),('amd','amd'),
]]
LOWPRI = {'nvidia','amd'}
def akey(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def brand_from_field(b):
    k = akey(b)
    if not k: return None, 'empty'
    if k in ALIAS: return ALIAS[k], 'exact'
    best = difflib.get_close_matches(k, list(ALIAS), n=1, cutoff=0.75)
    if best and len(k)>=4: return ALIAS[best[0]], 'fuzzy'
    return None, 'unmapped'
def brands_in_text(t):
    t = str(t).lower(); hits=[]
    for k,p in TITLE_BRAND_PAT:
        m = p.search(t)
        if m: hits.append((m.start(), k))
    hits.sort()
    out=[]
    for _,k in hits:
        if k not in out: out.append(k)
    return out

def parse_num(s):
    """Locale-robust number parse. Returns float or None."""
    s = str(s).strip().replace(' ',' ')
    if not s: return None
    s = s.replace(' ', '')
    if not re.fullmatch(r'[0-9.,]+', s) or not re.search(r'\d', s): return None
    if ',' in s and '.' in s:
        dec = ',' if s.rfind(',') > s.rfind('.') else '.'
        th = '.' if dec==',' else ','
        s = s.replace(th,'').replace(dec,'.')
    elif ',' in s:
        parts = s.split(',')
        if len(parts)==2 and len(parts[1])!=3: s = s.replace(',','.')
        elif len(parts)==2 and len(parts[1])==3 and parts[0]!='0': s = s.replace(',','')  # thousands
        elif len(parts)==2: s = s.replace(',','.')
        else: s = s.replace(',','')
    elif s.count('.')>1:
        s = s.replace('.','')
    elif '.' in s:
        p = s.split('.')
        if len(p[1])==3 and p[0] not in ('0',) and len(p[0])<=3 and not s.endswith('.0'):
            # "1.500" style thousands (only when exactly 3 decimals) -- ambiguous; treated as thousands
            s = s.replace('.','')
    if s.endswith('.'): s = s[:-1]
    try: return float(s)
    except: return None

PTYPE_ALIAS = {'ssd':'SSD','solidstatedrive':'SSD','nvmessd':'SSD','externalssd':'SSD','sataSSD'.lower():'SSD','m2ssd':'SSD','55d':'SSD','s5d':'SSD','5sd':'SSD','ss':'SSD','ssd_':'SSD',
 'hdd':'HDD','harddiskdrive':'HDD','externalhdd':'HDD','35inchhdd':'HDD','25inchhdd':'HDD','hd':'HDD','hdd_':'HDD','harddrive':'HDD','h':'HDD','hd0':'HDD',
 'gpu':'GPU','graphicsprocessingunit':'GPU','6pu':'GPU','gp':'GPU','gamingGPU'.lower():'GPU','graphicscard':'GPU','gpv':'GPU','g':'GPU','6p':'GPU',
 'usbstick':'USB_STICK','usbflashdrive':'USB_STICK','usbthumbdrive':'USB_STICK','usbst':'USB_STICK','flashdrive':'USB_STICK','usb':'USB_STICK'}
def ptype_from_field(p):
    k = akey(p)
    if not k: return None
    if k in PTYPE_ALIAS: return PTYPE_ALIAS[k]
    if k.startswith('usb') and len(k)>=5: return 'USB_STICK'
    return None
GPU_KW = re.compile(r'geforce|radeon|\bgtx\b|\brtx\b|\bgt ?\d{3,4}\b|graphics card|video card|grafikkort|videokaart|placa de v|quadro|\brx ?\d{3,4}|gddr|vga', re.I)
USB_KW = re.compile(r'flash drive|usb stick|pen ?drive|thumb drive|memory stick|datatraveler|\bcruzer|\bdt ?\w*|jumpdrive|usb flash|flash disk|clé usb|pendrive|ultra fit|ultra flair|ultra dual', re.I)
SSD_KW = re.compile(r'\bssd\b|solid state|nvme|\bm\.?2\b|msata|ssdnow', re.I)
HDD_KW = re.compile(r'\bhdd\b|hard (disk )?drive|harddisk|disco duro|\brpm\b|\b\d{4,5} ?rpm|barracuda|ironwolf|wd (red|blue|purple|gold|black)|exos|skyhawk|\bsas\b|dysk|festplatte|7\.2k|10k|15k|canvio|storejet|my passport|my book|expansion', re.I)
def ptype_from_text(t):
    t = str(t)
    if GPU_KW.search(t): return 'GPU'
    ssd = bool(SSD_KW.search(t)); hdd = bool(HDD_KW.search(t))
    if re.search(r'flash drive|usb stick|pen ?drive|thumb drive|datatraveler|cruzer|usb flash|flash disk|pendrive', t, re.I): return 'USB_STICK'
    if ssd and not hdd: return 'SSD'
    if hdd and not ssd: return 'HDD'
    if ssd and hdd:
        return 'SSD' if re.search(r'\bssd\b|solid state', t, re.I) else 'HDD'
    return None

CAP_RE = re.compile(r'(?<![0-9a-z.,])(\d{1,5}(?:[.,]\d{1,2})?)\s?-?(tb|gb|g|t|to|go|gigabyte|terabyte)(?:u\.2)?(?![a-z0-9])', re.I)
def capacities(text):
    out=[]
    for m in CAP_RE.finditer(str(text)):
        v = float(m.group(1).replace(',','.')); u = m.group(2).lower()
        # 'G' after an interface speed (3G/6G/12G SAS/SATA) is not a capacity
        if u=='g' and v in (1.5,3,6,12) and re.match(r'\s*(sas|sata|dp|sp|\d)', str(text)[m.end():], re.I): continue
        if re.match(r'\s*(/\s?s|ps|it/s|bps)', str(text)[m.end():], re.I): continue  # data rates (6Gb/s)
        if u=='t' and v in (3,4) and re.match(r'\s*(portable|type)', str(text)[m.end():], re.I): continue
        gb = v*1000 if u in ('tb','t','to','terabyte') else v
        out.append((m.start(), gb, u))
    return out

GPU_CHIP_RE = re.compile(r'(?<![a-z0-9])(rtx|gtx|gt|rx|r7|r9|r5|gts|quadro\s*(?:rtx\s*)?[a-z]?|vega|radeon\s*vii|titan|wx|radeon\s*pro\s*wx|t)\s?-?(\d{2,4})(?![0-9])\s*(ti|super|xt|s)?(?![a-z])\s*(super|xt)?(?![a-z])', re.I)
def gpu_chip(text):
    t = str(text).replace('®',' ').replace('™',' ')
    t = re.sub(r'(?i)geforce|nvidia|amd|radeon(?!\s*(vii|pro))', ' ', t)
    for m in GPU_CHIP_RE.finditer(t):
        ser = re.sub(r'\s+','',m.group(1).lower()); num = m.group(2)
        if ser=='t' and num not in ('400','600','1000','1200'): continue
        if ser.startswith('radeonpro'): ser='wx'
        if ser.startswith('quadro'):
            rest = ser[6:]
            ser = 'quadro' + (' rtx' if rest.startswith('rtx') else '') ; num = (rest[3:] if rest.startswith('rtx') else rest) + num
        suf = [ (m.group(3) or '').lower(), (m.group(4) or '').lower()]
        suf = ['super' if x=='s' else x for x in suf if x]
        if ser=='gt' and len(num)==4 and num not in ('1030','1010'): ser='gtx'
        if ser=='gtx' and num=='1030': ser='gt'
        return ' '.join([ser, num]+suf).strip()
    for v in ('vega 56','vega 64','radeon vii','titan rtx'):
        if v in t.lower(): return v
    return None
VRAM_RE = re.compile(r'(?<![0-9a-z])[oa]?(\d{1,2})\s?(gb|g)(?![a-z0-9])', re.I)
def vram_from(text):
    for m in VRAM_RE.finditer(str(text)):
        v = int(m.group(1))
        if v in (1,2,3,4,5,6,8,10,11,12,16,20,24,32,48): return float(v)
    return None
STOP_CODE = re.compile(r'^(gddr\d+x?|ddr\d|pcie\d*|usb\d+|sata\d*|\d+(gb|tb|g|t|mb|mhz|rpm|bit|mm|w)|\d+x\d+|m2|nvme|rtx\d+\w*|gtx\d+\w*|gt\d+|rx\d+\w*|2280|\d+)$')
def codes(*texts):
    out=set()
    for t in texts:
        for tok in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-/_.]{4,}[A-Za-z0-9]', str(t)):
            k = akey(tok)
            if len(k)<6 or not re.search(r'\d',k) or not re.search(r'[a-z]',k): continue
            if STOP_CODE.match(k): continue
            out.add(k)
    return out
# product-line cues -> brand (used only when no brand name is present in title/field)
LINE_PAT = [(k, re.compile(p, re.I)) for k,p in [
 ('wd',r'my passport|my book|\bsn5[05]0\b|\bsn7[05]0\b|\bwd\d|\bwds\d|\bred (ssd|pro|plus)|\bred\b.*\bwd\d|caviar|wd10|wd20|elements'),
 ('samsung',r'\bt[57] portable|portable ssd t[57]|\b8[67]0 (evo|pro|qvo)|\b9[678]0 (evo|pro)|\bm3 portable|v-nand|\bmz-'),
 ('kingston',r'\bkc[12][05]00\b|\bdc500|\ba400\b|\buv500|datatraveler|data traveler|\bsa400|\bsuv\d'),
 ('seagate',r'nytro|constellation|enterprise capacity|backup plus|game drive|archive hdd|\bst\d{3,5}[a-z]{2}|\bstel\d|\bstgx|\bstfc|\bstj'),
 ('msi',r'ventus|gaming x trio|\barmor\b'), ('gainward',r'\bphoenix\b'),
 ('adata',r'\bsx[68]\d00|\bsu[68]\d0|gammix|spectrix|\basu\d'), ('amd',r'radeon pro'),
 ('crucial',r'\bp[12] 2280|\bmx500|\bbx500|\bct\d{3,4}'), ('intel',r'\bdc s4[56]\d0|optane|\bdc p4\d{3}|\bssdpe|\bssdsc'),
 ('evga',r'\bxc black|\bxc gaming|\bftw3|\bsc ultra|\bxc ultra|\b\d{2}g-p\d-\d{4}'), ('asus',r'\bvgagt|rog-strix|\btuf-|\bdual-(rtx|gtx)'),
 ('hp',r'-b21\b'),
]]
def brand_from_lines(t):
    for k,p in LINE_PAT:
        if p.search(str(t)): return k
    return None
