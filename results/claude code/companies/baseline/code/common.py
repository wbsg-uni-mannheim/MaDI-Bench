import re, unicodedata
LEGAL = ['incorporated','inc','corporation','corp','company','co','ltd','limited','llc','l l c','plc','p l c','sa','s a','ag','se','gmbh','nv','n v','bv','b v','spa','s p a','sarl','sas','ab','asa','oyj','oy','as','a s','kgaa','kg','lp','l p','llp','pte','pty','bhd','berhad','tbk','pt','sab de cv','sab','de cv','cv','s a b','sa de cv','ltda','srl','s r l','nl','kk','k k','the','publ','public','pcl','tas','t a s','as','jsc','ojsc','pjsc','oao','pao','zao','ad','dd','d d','a g','s e','s p a','ag co','co ltd','holding ag']
LEGAL_SET = set(LEGAL)
WEAK = {'group','holdings','holding','the','international','intl','&','and','of','company','companies','corporation','bancorp','industries','enterprises'}

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def basic(s):
    s = strip_accents(str(s)).lower().replace('ı', 'i').replace('ø', 'o').replace('æ', 'ae').replace('ß', 'ss').replace('ł', 'l')
    s = s.replace('&', ' and ')
    s = re.sub(r"[’'`]", '', s)
    s = re.sub(r'[\W_]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def drop_parens(s):
    return re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', ' ', str(s)).strip()

def paren_content(s):
    return [p.strip() for p in re.findall(r'[\(\[]([^\)\]]*)[\)\]]', str(s)) if p.strip()]

def strip_legal_tokens(toks):
    # remove legal-form tokens at the end (repeatedly) and leading 'the'
    changed = True
    while changed and toks:
        changed = False
        for n in (3, 2, 1):
            if len(toks) > n and ' '.join(toks[-n:]) in LEGAL_SET:
                toks = toks[:-n]; changed = True; break
    if len(toks) > 1 and toks[0] == 'the':
        toks = toks[1:]
    return toks

def name_key(s):
    """strict matching key: accents/punct removed, parenthetical removed, trailing legal forms removed"""
    t = basic(drop_parens(s)).split()
    return ' '.join(strip_legal_tokens(t))

def core_key(s):
    """looser key: also removes generic corporate words"""
    t = [w for w in name_key(s).split() if w not in WEAK]
    return ' '.join(t) if t else name_key(s)

LEGAL_DISPLAY = re.compile(r'(,?\s+(Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?,?\s*Ltd\.?|Co\.?|Company|Ltd\.?|Limited|LLC|L\.L\.C\.|PLC|Plc|plc|P\.L\.C\.|S\.A\.|SA|AG|SE|GmbH|N\.V\.|NV|B\.V\.|BV|S\.p\.A\.|SpA|SARL|SAS|AB|ASA|Oyj|KGaA|LP|L\.P\.|LLP|Pty\.?|Pte\.?|Bhd\.?|Berhad|Tbk|S\.A\.B\. de C\.V\.|SAB de CV|de C\.V\.|Ltda\.?|S\.r\.l\.|\(publ\)|PCL|JSC|OJSC|PJSC))+\.?\s*$')

def display_name(s):
    s = str(s).strip()
    s = re.sub(r'\s*\((DEL|publ|The)\)\s*$', '', s, flags=re.I)
    prev = None
    while prev != s:
        prev = s
        s = LEGAL_DISPLAY.sub('', s).strip().rstrip(',').strip()
    s = re.sub(r'^The\s+(?=\S+\s+\S)', '', s)
    return s or prev
