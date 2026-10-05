"""Stage 2: normalization into a common canonical table per source (raw values kept)."""
import pandas as pd, numpy as np, json, os, re, ast, sys
from urllib.parse import unquote
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'work')
from common import *
from industry_map import FORBES, DBP_RULES

cldr = pd.read_csv('task/input/schemamatching/CLDR_Country_Taxonomy.csv')
gics = set(pd.read_csv('task/input/schemamatching/GICS_Industry_Taxonomy.csv')['Industry Name'])
CMAP = {}
for _, r in cldr.iterrows():
    for c in ['Country Name', 'Country Short Name', 'Country Variant Name']:
        if pd.notna(r[c]): CMAP[basic(r[c])] = r['Country Name']
EXTRA = {  # source spellings -> CLDR canonical (documented aliases)
 'england':'United Kingdom','scotland':'United Kingdom','wales':'United Kingdom','northern ireland':'United Kingdom',
 'united kingdom of great britain and northern ireland':'United Kingdom','peerage of the united kingdom':'United Kingdom',
 'kingdom of england':'United Kingdom','british empire':'United Kingdom',
 'united states of america':'United States','banking in the united states':'United States',
 'republic of ireland':'Ireland','georgia country':'Georgia','bosnia and herzegovina':'Bosnia & Herzegovina',
 'republika srpska':'Bosnia & Herzegovina','burma':'Myanmar (Burma)','republic of macedonia':'North Macedonia',
 'democratic republic of the congo':'Congo - Kinshasa','republic of the congo':'Congo - Brazzaville',
 'korea republic of':'South Korea','korea north':'North Korea','russian federation':'Russia',
 'taiwan province of china a':'Taiwan','venezuela bolivarian republic of':'Venezuela','the bahamas':'Bahamas',
 'kingdom of portugal':'Portugal','portuguese empire':'Portugal','west germany':'Germany','german empire':'Germany',
 'wurttemberg':'Germany','empire of japan':'Japan','grand duchy of finland':'Finland','upper canada':'Canada',
 'dutch republic':'Netherlands','british raj':'India','soviet union':'Russia','svalbard':'Svalbard & Jan Mayen',
}
def country(v):
    if pd.isna(v): return None
    k = basic(v)
    return CMAP.get(k) or (EXTRA.get(k) if EXTRA.get(k) in set(cldr['Country Name']) else None)

def dbp_industry(v):
    if pd.isna(v): return None
    s = str(v).lower()
    for pat, ind in DBP_RULES:
        if re.search(pat, s): return ind
    return None

def scale_money(v):
    # DBpedia monetary values carry no unit. Comparison against Forbes on exact-name pairs (see submission/report.md, Normalization)
    # shows values < 1e3 are reported in billions and 1e3..1e6 in millions. Scale accordingly; larger values are kept.
    if pd.isna(v) or v < 0: return np.nan
    if v < 1e3: return float(round(v * 1e9))
    if v < 1e6: return float(round(v * 1e6))
    return float(v)

def year(v):
    if pd.isna(v): return None
    m = re.match(r'^(\d{1,4})', str(v))
    if not m: return None
    y = int(m.group(1))
    return y if 1700 <= y <= 2016 else None   # schema consistency range

def people(v):
    if pd.isna(v): return None
    s = str(v).strip()
    if s.startswith('['):
        try: lst = ast.literal_eval(s)
        except Exception: lst = [x.strip(" '\"") for x in s.strip('[]').split(',')]
    else: lst = [s]
    out = []
    for x in lst:
        x = re.sub(r'\s+', ' ', str(x)).strip()
        if x and x not in out: out.append(x)
    return out or None

# ---- city extraction from DBpedia headquarters (concatenated location strings)
REGION_WORDS = set()
def build_regions(hq_series):
    reg = set(CMAP.keys()) | set(EXTRA.keys())
    from collections import Counter
    suffix, alone = Counter(), Counter()
    for v in hq_series.dropna():
        segs = split_hq(v)
        for seg in segs:
            if ',' in seg:
                suffix[basic(seg.split(',')[-1])] += 1
            else:
                alone[basic(seg)] += 1
    # a comma-suffix is treated as a region only if it is used more often as a suffix than as a stand-alone place
    reg |= {k for k, n in suffix.items() if n > alone.get(k, 0)}
    reg |= {basic(x) for x in ['England','Scotland','Wales','Maharashtra','Guangdong','Henan','Ontario','Quebec','British Columbia','Alberta','Bavaria','Tokyo','Gauteng','Karnataka','Texas','California','New York','Florida','Kanagawa','Osaka','Zhejiang','Jiangsu','Shandong','Hesse','Baden-Württemberg','North Rhine-Westphalia','Vaud','Zurich','Lombardy','Île-de-France','New South Wales','Victoria','Queensland','Delhi','Tamil Nadu','Gujarat','West Bengal','Telangana','Haryana','Uttar Pradesh','Kerala','Punjab','Sichuan','Hubei','Fujian','Liaoning','Hebei','Anhui','Hunan','Jilin','Shanxi','Shaanxi','Yunnan','Guangxi','Heilongjiang','Inner Mongolia','Jiangxi','Guizhou','Hainan','Xinjiang','Gansu','Ningxia','Qinghai','Tibet','Kyiv Oblast','Moscow Oblast','Dubai','Metro Manila','West Yorkshire','South Yorkshire','Greater London','Greater Manchester','Surrey','Kent','Essex','Berkshire','Hertfordshire','Middlesex','Hampshire','Merseyside','West Midlands','Lancashire','Cheshire','Buckinghamshire','Oxfordshire','Massachusetts','Pennsylvania','Illinois','Ohio','Michigan','Washington','Virginia','Georgia','New Jersey','Connecticut','Minnesota','Wisconsin','Colorado','Oregon','Arizona','Nevada','Utah','Tennessee','North Carolina','Missouri','Indiana','Maryland','Kentucky','Louisiana','Alabama','Oklahoma','Iowa','Kansas','Nebraska','Arkansas','Mississippi','South Carolina','Idaho','Maine','Vermont','New Hampshire','Rhode Island','Delaware','Hawaii','Alaska','Montana','Wyoming','North Dakota','South Dakota','New Mexico','West Virginia','District of Columbia']}
    reg -= {basic(x) for x in ['New York','Dubai','Tokyo','Singapore','Hong Kong','Delhi','Luxembourg','Monaco','Mexico','Kuwait','Panama','Djibouti','Victoria','Moscow','Zurich','Guatemala','Bermuda','Macau','Vatican City','Andorra','Osaka']}
    return reg
def split_hq(v):
    s = re.sub(r'([a-z\.\)ÿ-ž])([A-ZÀ-Þ])', r'\1|\2', str(v))
    return [x.strip() for x in s.split('|') if x.strip()]
def city_from_hq(v, regions):
    if pd.isna(v): return None
    segs = split_hq(v)
    cands = []
    for seg in segs:
        first = seg.split(',')[0].strip()
        first = re.sub(r'\s*\([^)]*\)', '', first).strip()
        if first and basic(first) not in regions and not re.search(r'\d|road|street|avenue|building|tower|plaza|quay|estate|campus|headquarters|centre|center|district|county|province|prefecture|region', first, re.I):
            cands.append(first)
    if cands:
        c = cands[0]
        return 'New York' if c in ('New York City', 'NYC') else c
    return None

REGIONISH = set('americas america north south east west europe asia emea english pacific latin middle international hr'.split())
def alias_keys(n):
    """alias name keys: parenthetical content and ' - '-separated segments, unless the dropped part is a region
    or the alias describes a parent ('a X company', 'subsidiary', 'division')"""
    n = str(n); out = []
    for p in paren_content(n):
        if not re.search(r'subsidiary|division|former|formerly|part of|a\.k\.a|\bdel\b|\bpte\b|\bpublic\b', p, re.I):
            k = name_key(p)
            if len(k) >= 3: out.append(k)
    segs = [x.strip() for x in re.split(r'\s+-\s*|\s*-\s+', drop_parens(n)) if x.strip()]
    if len(segs) > 1:
        cty = set()
        for v in CMAP: cty |= set(v.split())
        for i, sgm in enumerate(segs):
            rest = ' '.join(segs[:i] + segs[i+1:])
            rt = set(basic(rest).split())
            if rt and rt <= (cty | REGIONISH): continue
            if re.match(r'^(a|an)\s', sgm, re.I) or re.search(r'division|subsidiary', sgm, re.I): continue
            k = name_key(sgm)
            if len(k) >= 2: out.append(k)
    return out

def main():
    d = pd.read_csv('task/input/data/dbpedia.csv'); f = pd.read_csv('task/input/data/forbes.csv'); fc = pd.read_csv('task/input/data/fullcontact.csv')
    regions = build_regions(d.headquarters)
    rows = []
    for _, r in d.iterrows():
        nm = r.org_name
        uri_name = unquote(r.entity_uri.rsplit('/', 1)[-1]).replace('_', ' ')
        if '�' in str(nm): nm = uri_name
        rows.append(dict(source='dbpedia', id=r.entity_uri, name_raw=r.org_name, name=nm, alt=uri_name,
            country=country(r.nation), country_raw=r.nation, city=city_from_hq(r.headquarters, regions), city_raw=r.headquarters,
            industry=dbp_industry(r.sector), industry_raw=r.sector, founded=year(r.established), founded_raw=r.established,
            keypeople=people(r.keypeople_name), assets=scale_money(r.total_assets_val), revenue=scale_money(r.annual_income),
            assets_raw=r.total_assets_val, revenue_raw=r.annual_income))
    for _, r in f.iterrows():
        slug = r.forbes_url.rstrip('/').rsplit('/', 1)[-1].replace('-', ' ')
        rows.append(dict(source='forbes', id=r.forbes_url, name_raw=r.company, name=r.company, alt=slug,
            country=country(r.region), country_raw=r.region, city=None, city_raw=None,
            industry=FORBES.get(r.business_segment), industry_raw=r.business_segment, founded=None, founded_raw=None,
            keypeople=None, assets=r.asset_value, revenue=r.sales_figure))
    for _, r in fc.iterrows():
        rows.append(dict(source='fullcontact', id=r.Attribute_1, name_raw=r.Attribute_2, name=r.Attribute_2, alt=None,
            country=country(r.Attribute_3), country_raw=r.Attribute_3, city=(None if pd.isna(r.Attribute_4) else str(r.Attribute_4).strip()), city_raw=r.Attribute_4,
            industry=None, industry_raw=None, founded=year(r.Attribute_6), founded_raw=r.Attribute_6,
            keypeople=people(r.Attribute_5), assets=np.nan, revenue=np.nan))
    df = pd.DataFrame(rows)
    df['display'] = df.name.map(display_name)
    df['nkey'] = df.name.map(name_key); df['ckey'] = df.name.map(core_key)
    df['paren'] = df.name.map(lambda s: '|'.join(name_key(p) for p in paren_content(s)))
    df['alt_key'] = df.alt.map(lambda s: None if s is None or pd.isna(s) else name_key(s))
    # DBpedia parentheses are Wikipedia disambiguators (e.g. '(Israel)', '(Inditex)'), not aliases
    df['akeys'] = [alias_keys(n) if src != 'dbpedia' else [] for n, src in zip(df.name, df.source)]
    assert set(df.industry.dropna()) <= gics
    df['keypeople'] = df.keypeople.map(lambda x: json.dumps(x, ensure_ascii=False) if x else None)
    df.to_pickle('work/state/norm.pkl'); df.to_csv('work/state/norm.csv', index=False)
    # coverage
    cov = []
    for (s, a, raw) in [('dbpedia','country','country_raw'),('forbes','country','country_raw'),('fullcontact','country','country_raw'),
                        ('dbpedia','industry','industry_raw'),('forbes','industry','industry_raw'),('dbpedia','city','city_raw'),
                        ('dbpedia','founded','founded_raw'),('fullcontact','founded','founded_raw')]:
        x = df[df.source == s]; nn = x[raw].notna().sum(); cn = x[a].notna().sum()
        cov.append(dict(source=s, attribute=a, non_null=nn, canonical=cn, unmapped=nn-cn, canonical_rate=round(cn/max(nn,1),4)))
    cov = pd.DataFrame(cov); cov.to_csv('work/taxonomy_coverage.csv', index=False); print(cov)
    json.dump({'country': {'path': 'task/input/schemamatching/CLDR_Country_Taxonomy.csv', 'canonical': 'Country Name', 'aliases': ['Country Short Name','Country Variant Name'], 'exhaustive': True, 'extra_aliases': EXTRA, 'unmapped_policy': 'null'},
               'industry': {'path': 'task/input/schemamatching/GICS_Industry_Taxonomy.csv', 'canonical': 'Industry Name', 'exhaustive': True, 'forbes_table': 'work/industry_map.py:FORBES', 'dbpedia_rules': 'work/industry_map.py:DBP_RULES', 'unmapped_policy': 'null'},
               'keypeople': {'serialization': 'JSON list'}}, open('work/taxonomy_plan.json', 'w'), indent=1, ensure_ascii=False)
if __name__ == '__main__': main()
