"""Stage 2: normalization. Keeps raw values (raw_*) and adds canonical + comparison copies."""
import os, re, html, json, unicodedata, pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
T = pd.read_pickle('work/state/s1_translated.pkl')
TYPES = json.load(open('task/input/schemamatching/target_schema.json'))['properties']['type']['enum']
NULLS = {'', 'none', 'nan', 'null'}

def clean_str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if isinstance(v, float) and v.is_integer(): v = int(v)
    s = html.unescape(str(v)).strip()
    s = re.sub(r'\s+', ' ', s)
    return None if s.lower() in NULLS else s

def fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch)).lower()

def title_clean(v, src):
    s = clean_str(v)
    if s is None: return None
    s = re.sub(r'<[^>]+>', '', s).strip()
    if src == 'dblp' and s.endswith('.') and not s.endswith('..'):
        s = s[:-1].rstrip()   # DBLP appends a terminal period to every title
    return s or None

def tokens(s):
    return re.findall(r'[a-z0-9]+', fold(s)) if s else []

def author_clean(a):
    a = clean_str(a)
    if a is None: return None
    a = re.sub(r'\s+\d{4}$', '', a)  # DBLP homonym suffix "Bing Li 0005"
    return a

def surname(a):
    t = tokens(a)
    return t[-1] if t else None

def authors_list(v):
    if v is None or (isinstance(v, float)) or isinstance(v, str): return []
    out = [author_clean(a) for a in v]
    return [a for a in out if a]

def page_clean(v):
    s = clean_str(v)
    return s

N = pd.DataFrame({'id': T['id'], 'source': T['source']})
for c in ['type','title','authors','publication_year','journal','volume','issue','first_page','last_page','referenced_works_count','cited_by_count']:
    N['raw_'+c] = T[c]
N['type'] = T['type'].map(clean_str).map(lambda x: x if x in TYPES else None)
N['title'] = [title_clean(v, s) for v, s in zip(T['title'], T['source'])]
N['authors'] = T['authors'].map(authors_list)
N['publication_year'] = pd.to_numeric(T['publication_year'], errors='coerce').astype('Int64')
N['journal'] = T['journal'].map(clean_str)
for c in ['volume','issue','first_page','last_page']:
    N[c] = T[c].map(page_clean)
for c in ['referenced_works_count','cited_by_count']:
    N[c] = pd.to_numeric(T[c], errors='coerce').astype('Int64')
# comparison copies
RETRACT = re.compile(r'^\s*(retracted( article)?( on [a-z]+ \d{1,2}, \d{4})?\s*[:.\-]\s*)', re.I)
cmp_title = N['title'].map(lambda s: RETRACT.sub('', s) if s else s)
N['retracted_prefix'] = N['title'].map(lambda s: bool(s and RETRACT.match(s)))
N['tkey'] = cmp_title.map(lambda s: ''.join(tokens(s)))
N['ttok'] = cmp_title.map(tokens)
N['surnames'] = N['authors'].map(lambda L: [x for x in (surname(a) for a in L) if x])
N['jtok'] = N['journal'].map(lambda s: ' '.join(tokens(s)) if s else None)
N.to_pickle('work/state/s2_normalized.pkl')
# coverage
rows=[]
for s,g in N.groupby('source'):
    for c in ['type','title','authors','publication_year','journal','volume','issue','first_page','last_page','referenced_works_count','cited_by_count']:
        raw = g['raw_'+c]
        nn_raw = raw.map(lambda v: clean_str(v) is not None if not isinstance(v, list) else len(v)>0).sum()
        can = g[c].map(lambda v: (len(v)>0) if isinstance(v,list) else pd.notna(v)).sum()
        rows.append(dict(source=s,attribute=c,non_null=int(nn_raw),canonical=int(can),unmapped=int(nn_raw-can),canonical_rate=round(can/max(nn_raw,1),4)))
pd.DataFrame(rows).to_csv('work/taxonomy_coverage.csv',index=False)
json.dump({'type':{'vocabulary':TYPES,'exhaustive':True,'aliases':{},'unmapped_policy':'null'},
           'authors':{'serialization':'JSON list','dblp_homonym_suffix':'stripped'},
           'title':{'html':'unescaped, tags stripped','dblp_trailing_period':'removed'}},
          open('work/taxonomy_plan.json','w'),indent=1)
print(pd.DataFrame(rows).query('unmapped!=0'))
