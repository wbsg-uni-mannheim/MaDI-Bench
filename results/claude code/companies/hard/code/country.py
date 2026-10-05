import re, unicodedata, pandas as pd, os
from rapidfuzz import fuzz, process
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAX = pd.read_csv(f'{ROOT}/task/input/schemamatching/CLDR_Country_Taxonomy.csv')
CANON = list(TAX['Country Name'])
ALIAS = {}
for _, r in TAX.iterrows():
    for c in ['Country Name', 'Country Short Name', 'Country Variant Name']:
        if isinstance(r[c], str): ALIAS[r[c]] = r['Country Name']
# manually curated aliases (official/native/historical names, constituent countries)
EXTRA = {
 'USA':'United States','United States of America':'United States','America':'United States','U.S.':'United States','U.S.A.':'United States',
 'Banking in the United States':'United States',
 'England':'United Kingdom','Scotland':'United Kingdom','Wales':'United Kingdom','Northern Ireland':'United Kingdom',
 'Great Britain':'United Kingdom','Britain':'United Kingdom','United Kingdom of Great Britain and Northern Ireland':'United Kingdom',
 'Kingdom of Great Britain and Northern Ireland':'United Kingdom','Kingdom of England':'United Kingdom','British Empire':'United Kingdom','Scotland, UK':'United Kingdom',
 'Isle of Man':'Isle of Man','Nippon':'Japan','Empire of Japan':'Japan','Russian Federation':'Russia','Россия':'Russia','Soviet Union':'Russia',
 'Korea (Republic of)':'South Korea','Republic of Korea':'South Korea','Korea':'South Korea','Korea South':'South Korea',
 'Korea, North':'North Korea','North Korea,':'North Korea',
 'Taiwan, Province of China[a]':'Taiwan','Taiwan, Province of China':'Taiwan','Republic of China':'Taiwan','Republic of China (1912–49)':'Taiwan',
 'Republic of India':'India','Bharat':'India','British Raj':'India','Republic of Ireland':'Ireland','República de Irlanda':'Ireland',
 "People's Republic of China":'China','中华人民共和国':'China','PRC':'China',
 'Italia':'Italy','Republic of Italy':'Italy','España':'Spain','Swiss Confederation':'Switzerland','Confederation Suisse':'Switzerland','Schweiz':'Switzerland',
 'Commonwealth of Australia':'Australia','Aotearoa':'New Zealand','Türkiye':'Turkey','UAE':'United Arab Emirates',
 'Dutch Kingdom':'Netherlands','Holland':'Netherlands','The Netherlands':'Netherlands','Kingdom of the Netherlands':'Netherlands',
 'HK':'Hong Kong SAR China','Georgia (country)':'Georgia','Republic of Georgia':'Georgia','Republik Österreich':'Austria','Österreich':'Austria',
 'State of Israel':'Israel','Republic of Poland':'Poland','Brasil':'Brazil','Brazilian Republic':'Brazil','Republika Srbija':'Serbia','Republic of Serbia':'Serbia',
 'Kingdom of Norway':'Norway','Bosnia and Herzegovina':'Bosnia & Herzegovina','BiH':'Bosnia & Herzegovina','Burma':'Myanmar (Burma)',
 'Republic of Macedonia':'North Macedonia','Macedonia':'North Macedonia','República Portuguesa':'Portugal','Kingdom of Portugal':'Portugal','Portuguese Empire':'Portugal','Império Português':'Portugal',
 'French Republic':'France','Republic of Singapore':'Singapore','共和国新加坡':'Singapore','Islamic Republic of Pakistan':'Pakistan',
 'Republic of the Philippines':'Philippines','República Argentina':'Argentina','Kingdom of Belgium':'Belgium','Republika Hrvatska':'Croatia',
 'República de Chile':'Chile','Islamic Republic of Iran':'Iran','Venezuela (Bolivarian Republic of)':'Venezuela','Bolivarian Republic of Venezuela':'Venezuela',
 'Azerbaijan Republic':'Azerbaijan','ราชอาณาจักรไทย':'Thailand','Republic of South Africa':'South Africa','Republik Indonesia':'Indonesia',
 'República de Colombia':'Colombia','Grand-Duché de Luxembourg':'Luxembourg','Crna Gora':'Montenegro','Republic of Kazakhstan':'Kazakhstan',
 'State of Kuwait':'Kuwait','KSA':'Saudi Arabia','Việt Nam':'Vietnam','SFR Yugoslavia':'Serbia','Grand Duchy of Finland':'Finland','Suomi':'Finland',
 'Estados Unidos Mexicanos':'Mexico','México':'Mexico','State of Qatar':'Qatar','German Empire':'Germany','Deutschland':'Germany','Württemberg':'Germany',
 'Republic of Peru':'Peru','Hellas':'Greece','Slovak':'Slovakia','Slovak Republic':'Slovakia','Czech':'Czechia','Republika Slovenija':'Slovenia',
 'Democratic Republic of the Congo':'Congo - Kinshasa','Bielorussie':'Belarus','Northern Cyprus':'Cyprus','Spanish protectorate in Morocco':'Morocco',
 'Guanxi, Hsinchu':'Taiwan','Mailiao, Yunlin':'Taiwan','Principality of Sealand':None,'European Union':None,'Other':None,'United Airlines':None,
 '<UUNCHANGED>':None,'<N UNCHANGED>':None,
 'Kingdom':'United Kingdom','Kong':'Hong Kong SAR China','Can':'Canada','Ja':'Japan','Sp':'Spain','Spa':'Spain','Gre':'Greece','of Man':'Isle of Man',
 'USA of America':'United States','United':None,'South':None,'New':None,'Ch':None,'Chi':None,'In':None,'Bu':None,'P':None,'Africa':None,'State':None,
 'Republic':None,'Republic of':None,'Republic of the':None,'Austr':None,
}
ALIAS.update({k:v for k,v in EXTRA.items()})
OCR = str.maketrans({'0':'o','1':'l','3':'e','4':'a','5':'s','6':'g','8':'b','9':'g','7':'t'})
def key(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode().lower()
    s = s.translate(OCR)
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'[^a-z ]', ' ', s)
    return ' '.join(sorted(s.split()))
def nospace(s): return key(s).replace(' ','')
KEYMAP = {}
for a, c in ALIAS.items():
    KEYMAP.setdefault(key(a), c)
KEYS = [k for k in KEYMAP if k]
NOSP = {}
for k in KEYS: NOSP.setdefault(''.join(sorted(k.replace(' ',''))), KEYMAP[k])
CONFMAP = str.maketrans({'o':'a','c':'e','h':'b','l':'i','j':'i','y':'i'})
def conf(k): return ' '.join(sorted(k.translate(CONFMAP).split()))
CONF = {}
for k in KEYS: CONF.setdefault(conf(k), KEYMAP[k])
_cache = {}
def resolve(raw):
    """returns (canonical or None, method)"""
    if not isinstance(raw, str) or not raw.strip(): return (None, 'missing')
    if raw in _cache: return _cache[raw]
    s = raw.strip()
    if s in ALIAS: out = (ALIAS[s], 'alias') if ALIAS[s] else (None,'unmappable')
    else:
        k = key(s)
        ck = conf(k)
        if k in KEYMAP: out = (KEYMAP[k], 'key')
        elif ck in CONF and len(k)>3: out = (CONF[ck], 'confusable')
        elif ''.join(sorted(k.replace(' ',''))) in NOSP and len(k)>3: out = (NOSP[''.join(sorted(k.replace(' ','')))], 'anagram')
        else:
            out = (None, 'unmapped')
            if len(k.replace(' ','')) >= 3:
                # prefix truncation (e.g. 'Canad', 'United Stat')
                ks = k.replace(' ','')
                pref = {KEYMAP[x] for x in KEYS if x.replace(' ','').startswith(ks) or ''.join(sorted(x.split())).startswith(ks)}
                raw_pref = {c for a,c in ALIAS.items() if c and a.lower().replace(' ','').startswith(s.lower().replace(' ',''))}
                cands = (pref | raw_pref) - {None}
                if len(cands) == 1 and len(ks) >= 3: out = (cands.pop(), 'prefix')
                else:
                    m = process.extractOne(k, KEYS, scorer=fuzz.token_sort_ratio)
                    m2 = process.extractOne(k, KEYS, scorer=fuzz.ratio)
                    best = max([m, m2], key=lambda x: x[1])
                    thr = 80 if len(ks) <= 6 else 72
                    if best[1] >= thr and KEYMAP[best[0]]:
                        out = (KEYMAP[best[0]], f'fuzzy{int(best[1])}')
    _cache[raw] = out
    return out
