"""Stage 2: normalization. Symmetric canonicalization of every source into target representation.
Raw values are kept next to normalized ones for audit."""
import pandas as pd, numpy as np, re, json, unicodedata, ast
from rapidfuzz import process, fuzz
U=pd.read_pickle('work/state/s1_unified_raw.pkl')

def fold(s): return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode()

# ---------------- name ----------------
LEGAL=r'(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.l\.c|plc|p\.l\.c|sa|s\.a|ag|se|nv|n\.v|bv|gmbh|sarl|sas|spa|s\.p\.a|ab|asa|oyj|a/s|as|kk|k\.k|bhd|berhad|tbk|pte|pty|lp|llp|group|holdings?)'
def clean_name(s):
    s=s.strip()
    s=re.sub(r'\s*\((company|corporation|firm|business|brand|retailer|bank|airline|conglomerate|[^)]*company[^)]*|[^)]*firm[^)]*)\)\s*$','',s,flags=re.I)
    s=re.sub(r'\s*\((DEL|DE|NEW|THE|NY|Pte|Pty)\)\s*$','',s,flags=re.I)
    # strip legal-form suffixes (repeatedly) but keep 'Group'/'Holdings' as part of the name
    lg=r'(Inc|Incorporated|Corp|Corporation|Co|Ltd|Limited|LLC|L\.L\.C|PLC|P\.L\.C|S\.?A|AG|SE|N\.?V|B\.?V|GmbH|SARL|SAS|S\.?p\.?A|AB|ASA|Oyj|A/S|K\.?K|Bhd|Berhad|Tbk|LP|LLP|S\.A\.B\. de C\.V|SAB de CV)'
    for _ in range(3):
        s2=re.sub(r'[,\s]+'+lg+r'\.?\s*$','',s,flags=re.I)
        s2=re.sub(r'[,\s]+(Co|Company)[.,]?\s*$','',s2,flags=re.I) if s2!=s else s2
        if s2==s: break
        s=s2.strip(' ,')
    return s.strip(' ,') or None
LEGALTOK={'inc','incorporated','corp','corporation','co','company','cos','ltd','limited','llc','plc','sa','ag','se','nv','bv',
 'gmbh','sarl','sas','spa','ab','asa','oyj','kk','bhd','berhad','tbk','pte','pty','lp','llp','sab','cv','del','oao','pao','ojsc','jsc','pjsc','ooo','zao','srl','sl','as','the','kgaa','sae','saa','nl','hr'}
LOOSETOK={'group','holding','holdings','companies','and','of','de','international','intl'}
def _toks(s):
    s=fold(s).lower().replace('&',' and ')
    s=re.sub(r"[’'`]",'',s); s=re.sub(r'[^a-z0-9]+',' ',s)
    return s.split()
GENERIC_PAREN=re.compile(r'^(company|corporation|firm|business|group|holding|holdings|enterprise|conglomerate|brand|international|official|[a-z]{1,4})$',re.I)
def pre_clean(raw):
    raw=re.sub(r'(?i)\.(com|net|org|co\.uk)\b','',raw)          # Via.com -> Via (domain suffix, not a word)
    r=re.sub(r'(?<=\b[A-Za-z])\.(?=\s*[A-Za-z]\b)','',raw)    # S.p.A. -> SpA ; J.C. -> JC
    r=re.sub(r'(?<=\b[A-Za-z])\.','',r)
    r=re.sub(r'\b([A-Za-z])/([A-Za-z])\b',r'\1\2',r)            # A/S -> AS
    parts=re.split(r'\s+[-|–]\s+',r)
    if len(parts)>1:                                              # "Salzgitter AG - HR", "IAI - Israel Aerospace"
        keep=[x for x in parts if not re.fullmatch(r'[A-Z0-9&]{1,6}',x.strip())]
        r=' '.join(keep) if keep else r
    return r
def name_keys(raw):
    raw=pre_clean(raw)
    paren=' '.join(re.findall(r'\((.*?)\)',raw))
    base=re.sub(r'\(.*?\)',' ',raw)
    toks=_toks(base)
    strict=[t for t in toks if t not in LEGALTOK] or toks
    loose=[t for t in strict if t not in LOOSETOK] or strict
    pc=None
    if paren:
        c,how=norm_country(paren.split(',')[0])
        if c and how in ('exact','perm'): pc=c
    if not toks:   # non-latin or empty after folding: use casefolded raw
        r=raw.casefold().strip(); return r,r,r,pc
    full=' '.join(toks)
    if pc: strict=strict+['@'+pc]; loose=loose+['@'+pc]
    return ' '.join(strict),' '.join(loose),full,pc
def paren_desc(raw):
    ps=re.findall(r'\((.*?)\)',raw)
    for p in ps:
        c,how=norm_country(p.split(',')[0])
        if c and how in ('exact','perm'): continue
        if GENERIC_PAREN.match(p.strip()): continue
        return p.strip()
    return None
# ---------------- founded ----------------
TR=str.maketrans({'O':'0','o':'0','p':'0','P':'0','D':'0','l':'1','I':'1','i':'1','q':'1','Q':'1','!':'1','w':'2','W':'2','Z':'2','z':'2','E':'3','e':'3','A':'4','S':'5','s':'5','G':'6','b':'6','T':'7','B':'8','g':'9'})
def parse_year(v):
    if not isinstance(v,str): return None
    v=v.strip().replace(' ','')
    m=re.match(r'^(\d\d)\.(\d\d)\.(.{4})$',v)      # fullcontact DD.MM.YYYY
    y=m.group(3) if m else v[:4]
    y=y.translate(TR)
    if re.fullmatch(r'\d{4}',y) and 1700<=int(y)<=2016: return int(y)
    return None
U['founded_year']=U.founded_raw.map(parse_year)
U['founded']=U.founded_year.map(lambda y: f'{int(y):04d}-01-01' if pd.notna(y) else None)

# ---------------- country ----------------
tax=pd.read_csv('task/input/schemamatching/CLDR_Country_Taxonomy.csv',dtype=str,keep_default_na=False)
CANON={}
def ck(s): return re.sub(r'[^a-z0-9]','',fold(s).lower())
for _,r in tax.iterrows():
    for c in ['Country Name','Country Short Name','Country Variant Name']:
        if r[c]: CANON[ck(r[c])]=r['Country Name']
ALIAS={'United States':['United States of America','USA','US of America','USA of America','America','States','Banking in the United States'],
 'United Kingdom':['United Kingdom of Great Britain and Northern Ireland','UK of Great Britain and Northern Ireland','England','Scotland','Wales','Northern Ireland','Great Britain','Britain','England, UK','Kingdom of England','Peerage of the United Kingdom','British Empire'],
 'South Korea':['Korea (Republic of)','Republic of Korea','Korea South','Korea','대한민국'],'North Korea':['Korea, North'],
 'Taiwan':['Taiwan, Province of China[a]','Taiwan Republic of China','Province of China[a]','Taiwan, of China[a]'],
 'Ireland':['Republic of Ireland'],'Russia':['Russian Federation','Soviet Union','Russian'],'China':["People's Republic of China",'PRC'],
 'India':['Republic of India','British Raj'],'Georgia':['Georgia (country)'],'Bosnia & Herzegovina':['Bosnia and Herzegovina','Republika Srpska'],
 'Myanmar (Burma)':['Burma'],'Australia':['Commonwealth of Australia'],'North Macedonia':['Republic of Macedonia','Macedonia'],
 'New Zealand':['Zealand New','Aotearoa'],'Hong Kong SAR China':['Kong Hong','Hongkong'],'Italy':['Repubblica Italiana','Italian Republic'],
 'United Arab Emirates':['UAE','Arab Emirates','United Emirates','United Emirates Arab'],'Congo - Kinshasa':['Democratic Republic of the Congo'],
 'Congo - Brazzaville':['Republic of the Congo'],'Singapore':['Republic of Singapore'],'Philippines':['Republic of the Philippines','Republika ng Pilipinas'],
 'Portugal':['Portuguese Empire','Kingdom of Portugal','Portugal of Kingdom'],'Venezuela':['Venezuela (Bolivarian Republic of)'],'Finland':['Suomi','Grand Duchy of Finland'],
 'Czechia':['Czech','Republic Czech'],'Belgium':['Kingdom of Belgium'],'Croatia':['Hrvatska'],'Japan':['Japan, Inc.','Nippon','Empire of Japan'],
 'Estonia':['Estonian Republic'],'Israel':['State of Israel'],'Germany':['German Empire','Württemberg'],'South Africa':['RSA','Africa South'],
 'Saudi Arabia':['KSA'],'Bahamas':['The Bahamas'],'Spain':['España'],'Uruguay':['República Oriental del Uruguay'],'Sudan':['Republic of the Sudan']}
for can,al in ALIAS.items():
    assert ck(can) in CANON, can
    for a in al: CANON[ck(a)]=CANON[ck(can)]
# token-permutation aliases ("States United", "Kingdom United" ...)
TOKCANON={}
for k_raw in list(tax['Country Name'])+[a for al in ALIAS.values() for a in al]+list(ALIAS):
    t=' '.join(sorted(re.sub(r'[^a-z ]',' ',fold(k_raw).lower()).split()))
    if t: TOKCANON.setdefault(t, CANON.get(ck(k_raw)))
KEYS=list(CANON)
def norm_country(v):
    if not isinstance(v,str) or not v.strip(): return None,'null'
    k=ck(v)
    if k in CANON: return CANON[k],'exact'
    t=' '.join(sorted(re.sub(r'[^a-z ]',' ',fold(v).lower()).split()))
    if t in TOKCANON and TOKCANON[t]: return TOKCANON[t],'perm'
    if len(k)>=5:
        m=process.extractOne(k,KEYS,scorer=fuzz.ratio)
        if m and m[1]>=80: return CANON[m[0]],'fuzzy'
    return None,'unmapped'
cc=U.country_raw.map(norm_country)
U['country']=cc.map(lambda x:x[0]); U['country_how']=cc.map(lambda x:x[1])

U['name']=U.name_raw.map(lambda v: clean_name(v) if isinstance(v,str) else None)
nk=U.name_raw.map(name_keys)
U['k']=nk.map(lambda x:x[0]); U['k_loose']=nk.map(lambda x:x[1]); U['k_full']=nk.map(lambda x:x[2]); U['paren_country']=nk.map(lambda x:x[3])
U['paren_desc']=U.name_raw.map(paren_desc)
# ---------------- city ----------------
def norm_city(v):
    if not isinstance(v,str): return None
    v=v.strip()
    v=re.sub(r'([a-zà-ÿ])([A-Z])',r'\1|\2',v)   # "SeoulIncheon" -> first value
    v=v.split('|')[0].split(',')[0].strip()
    v=re.sub(r'\s*\(.*?\)','',v).strip()
    if not v or re.fullmatch(r'[\W\d_]+',v): return None
    return v
U['city']=U.city_raw.map(norm_city)

# ---------------- industry (GICS, exhaustive) ----------------
g=pd.read_csv('task/input/schemamatching/GICS_Industry_Taxonomy.csv',dtype=str)
INDS=set(g['Industry Name'])
OVR=json.load(open('work/industry_overrides.json'))
C=json.load(open('work/state/emb_cache.json'))
labs=[(r['Industry Name'],r['Industry Name']) for _,r in g.drop_duplicates('Industry Name').iterrows()]+[(r['Sub-Industry Name'],r['Industry Name']) for _,r in g.iterrows()]
M=np.array([C[t] for t,_ in labs]); M/=np.linalg.norm(M,axis=1,keepdims=True)
def norm_ind(v):
    if not isinstance(v,str) or not v.strip(): return None,'null',None
    v=v.strip()
    if v in OVR: return OVR[v],'override',1.0
    if v in INDS: return v,'exact',1.0
    e=np.array(C[v]); e/=np.linalg.norm(e); s=M@e; i=int(s.argmax())
    if s[i]>=0.40: return labs[i][1],'embed',float(s[i])
    return None,'unmapped',float(s[i])
ii=U.industry_raw.map(norm_ind)
U['industry']=ii.map(lambda x:x[0]); U['industry_how']=ii.map(lambda x:x[1])
assert set(U.industry.dropna())<=INDS

# ---------------- money ----------------
MULT={'thousand':1e3,'k':1e3,'million':1e6,'millipn':1e6,'mn':1e6,'m':1e6,'mio':1e6,'billion':1e9,'bn':1e9,'b':1e9,'trillion':1e12,'tn':1e12}
def parse_num(tok):
    tok=tok.strip()
    if re.fullmatch(r'\d{1,3}(\.\d{3})+,\d+',tok): return float(tok.replace('.','').replace(',','.'))   # 4.796.000.000,0
    if re.fullmatch(r'\d{1,3}(\.\d{3}){2,}',tok): return float(tok.replace('.',''))
    if re.fullmatch(r'\d{1,3}(,\d{3})+(\.\d+)?',tok): return float(tok.replace(',',''))
    if re.fullmatch(r'\d+(\.\d+)?',tok): return float(tok)
    if re.fullmatch(r'\d+,\d+',tok): return float(tok.replace(',','.'))
    return None
def parse_money(v,src,attr):
    """returns (value_usd_face, how). Currency symbols are NOT converted (face value kept; see report)."""
    if not isinstance(v,str) or not v.strip(): return None,'null'
    s=v.strip()
    s=re.sub(r'\s+(USD|US\$|EUR|GBP)$','',s)
    m=re.fullmatch(r'(?:[A-Z]{1,3}\$|[A-Z]{3}|\$|€|£|¥)?\s*([\d.,]+)\s*([A-Za-z]+)?\.?',s)
    if not m: return None,'unparsed_text'
    num=parse_num(m.group(1)); unit=(m.group(2) or '').lower()
    if num is None: return None,'unparsed_num'
    if unit:
        if unit not in MULT:
            fm=process.extractOne(unit,['million','billion','trillion','thousand'],scorer=fuzz.ratio)
            if fm and fm[1]>=80: unit=fm[0]
            else: return None,'unparsed_unit'
        return num*MULT[unit],'unit'
    bare=re.fullmatch(r'[\d.]+',m.group(1)) is not None and s[0].isdigit()
    if src=='forbes' and attr=='revenue' and bare and num<5000: return num*1e9,'forbes_billions'
    if src=='dbpedia' and bare and num<100: return num*1e9,'small_as_billions'
    if src=='dbpedia' and bare and num<1e5: return num*1e6,'mid_as_millions'
    return num,'plain'
for a in ['assets','revenue']:
    pm=U.apply(lambda r: parse_money(r[a+'_raw'],r.source,a),axis=1)
    U[a]=pm.map(lambda x: int(round(x[0])) if x[0] is not None else None).astype('Int64')
    U[a+'_how']=pm.map(lambda x:x[1])
    lim={'assets':4e12,'revenue':1e12}[a]
    bad=U[a].notna()&((U[a]<0)|(U[a]>lim))
    U.loc[bad,a]=pd.NA; U.loc[bad,a+'_how']='out_of_range'

# ---------------- keypeople ----------------
GENERIC=re.compile(r'\b(team|board|committee|executive|executives|management|leadership|unavailable|unknown|founders?|led by|partners|directors?|staff|government|n/a|none|group)\b',re.I)
ROLE=re.compile(r'\b(chief [a-z ]*officer|ceo|cfo|cto|coo|founder-ceo|co-founder|cofounder|founder|founding [a-z]+|chair(man|woman|person)?|board chair|president|managing director|executive director|general manager|creative director|technical lead|director|chief scientist|operations director|producer|systems architect|partner|serves as|while|is chair|md)\b',re.I)
def clean_person(p):
    p=p.strip().strip('"\'[]').strip()
    p=re.sub(r'\(.*?\)','',p)
    if ':' in p: p=p.split(':',1)[1]
    p=ROLE.sub(' ',p)
    p=re.sub(r'\bDr\.\s*','',p)
    p=' '.join(p.replace(',',' ').split()).strip(' .;-')
    if not p or len(p)<3: return None
    if GENERIC.search(p): return None
    if not re.search(r'[A-Za-zÀ-ɏ]',p): return None
    return p
def parse_people(v):
    if not isinstance(v,str) or not v.strip(): return None
    v=v.strip()
    if v.startswith('['):
        try: items=[str(x) for x in ast.literal_eval(v)]
        except Exception: items=re.split(r"',\s*'",v)
    else:
        items=re.split(r';|\band\b(?=\s+[A-Z])',v)
        out=[]
        for it in items:
            # "Name, Role" or "Name1, Name2"
            parts=[x for x in it.split(',')]
            if len(parts)>1 and all(ROLE.search(x) is None for x in parts) and all(len(x.split())>=2 for x in parts):
                out+=parts
            elif len(parts)==2 and ROLE.search(parts[1]) and not ROLE.search(parts[0]):
                out.append(parts[0])
            elif len(parts)==2 and ROLE.search(parts[1]) and len(parts[0].split())==1:  # "Cairn, Elspeth Managing Director"
                out.append(parts[1]+' '+parts[0])
            else: out.append(it)
        items=out
    ppl=[]
    for it in items:
        c=clean_person(it)
        if c and c not in ppl: ppl.append(c)
    return ppl or None
U['keypeople']=U.keypeople_raw.map(parse_people)

U.to_pickle('work/state/s2_normalized.pkl')
# coverage report
rows=[]
for (src),grp in U.groupby('source'):
    for a in ['founded','country','city','industry','assets','revenue','keypeople']:
        nn=grp[a+'_raw'].notna().sum(); can=grp[a].notna().sum()
        rows.append(dict(source=src,attribute=a,non_null=int(nn),canonical=int(can),unmapped=int(nn-can),canonical_rate=round(can/nn,3) if nn else None))
cov=pd.DataFrame(rows); cov.to_csv('work/taxonomy_coverage.csv',index=False); print(cov.to_string())
