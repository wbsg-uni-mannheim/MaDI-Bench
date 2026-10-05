import pandas as pd, re, json, sys
sys.path.insert(0,'work'); from common import *
IN='task/input/data/'
SUFFIX=re.compile(r'\s*\((album|release|orig\.)\)\s*$', re.I)
PREFIX=re.compile(r'^\s*(\*popular\*|\[explicit\])\s*', re.I)
COUNTRY={'UK':'United Kingdom of Great Britain and Northern Ireland','United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
 'Great Britain':'United Kingdom of Great Britain and Northern Ireland','US':'United States of America','USA':'United States of America',
 'CA':'Canada','NO':'Norway','NZ':'New Zealand','Russia':'Russian Federation','South Korea':'Korea, Republic of',
 'Bolivia':'Bolivia, Plurinational State of','Bosnia & Herzegovina':'Bosnia and Herzegovina','Moldova':'Moldova, Republic of',
 'Macedonia':'North Macedonia','Taiwan':'Taiwan, Province of China','Venezuela':'Venezuela, Bolivarian Republic of',
 'United Kingdom & Europe':'UK & Europe','Czech Republic':'Czechia'}
# keep Venezuela/Czech as MB spelled them in source (they appear verbatim in MB); only map genuine aliases
for k in ['Venezuela','Czech Republic','Taiwan','Macedonia']: COUNTRY.pop(k)
def norm_date(v):
    if not isinstance(v,str): return None, None
    s=re.sub(r'\s+','',v)
    if re.fullmatch(r'\d{8}',s.replace('-','')):
        s=s.replace('-',''); s=f'{s[:4]}-{s[4:6]}-{s[6:]}'
    return (s if re.fullmatch(r'\d{4}-\d{2}-\d{2}',s) else None), s
def norm_dur(v):
    if not isinstance(v,str) or not v.strip(): return None
    v=v.strip()
    if re.fullmatch(r'\d+',v):
        n=int(v); return n if n>0 else None   # 0 = unknown
    p=v.split(':')
    if all(x.isdigit() for x in p) and len(p) in (2,3):
        n=0
        for x in p: n=n*60+int(x)
        return n if n>0 else None
    return None
def clean_name(v, src):
    if not isinstance(v,str): return None
    s=fix_moji(v)
    s2=PREFIX.sub('',s)
    if s2.strip(): s=s2          # a bare tag is kept as the (only) title evidence
    if src=='musicbrainz': s=SUFFIX.sub('',s)
    s=re.sub(r'\s{2,}',' ',s).strip()
    return s
def clean_artist(v, src):
    if not isinstance(v,str): return None
    s=fix_moji(v).strip()
    return s
def artist_keys(a, src):
    """list of normalized artist alternatives (comparison only)"""
    if not a: return []
    parts=a.split('|') if src=='discogs' else [a]
    out=[]
    for p in parts:
        p=re.sub(r'\s*\(\d+\)\s*$','',p)             # discogs disambiguation "(2)"
        p=re.sub(r'\s+feat\.?.*$','',p,flags=re.I)
        if src=='musicbrainz' and p.count(',')==1:
            a1,b1=[x.strip() for x in p.split(',')]
            if b1.lower()=='the': p='The '+a1
            elif b1: p=b1+' '+a1
        out.append(key(p))
    return out
rows={}
cov=[]
for s in ['discogs','lastfm','musicbrainz']:
    x=pd.read_csv(IN+s+'.csv',dtype=str,keep_default_na=False,na_values=[''])
    o=pd.DataFrame({'id':x.id,'source':s})
    o['name_raw']=x.name; o['name']=[clean_name(v,s) for v in x.name]; o['name_key']=o.name.map(key)
    o['artist_raw']=x.artist; o['artist']=[clean_artist(v,s) for v in x.artist]
    o['artist_keys']=[json.dumps(artist_keys(v,s)) for v in o.artist]
    o['duration_raw']=x.duration; o['duration']=x.duration.map(norm_dur).astype('Int64')
    tr=[[fix_moji(t).strip() for t in parse_list(v)] for v in x.tracks]
    o['tracks']=[json.dumps([t for t in L if t]) for L in tr]
    o['tracks_key']=[json.dumps(sorted(set(key(t) for t in L if key(t)))) for L in tr]
    if 'release-date' in x:
        nd=[norm_date(v) for v in x['release-date']]
        o['date_raw']=x['release-date']; o['date']=[a for a,b in nd]
        o['country_raw']=x['release-country']; o['country']=x['release-country'].map(lambda v: COUNTRY.get(v.strip(),v.strip()) if isinstance(v,str) else None)
    else:
        o['date']=None;o['country']=None
    if 'label' in x:
        o['label']=x.label.map(lambda v: json.dumps([fix_moji(t).strip() for t in v.split('|') if t.strip()]) if isinstance(v,str) else '[]')
        o['genre']=x.genre.map(lambda v: v.strip() if isinstance(v,str) else None)
    else: o['label']='[]'; o['genre']=None
    for col,raw in [('duration','duration_raw'),('date','date_raw')]:
        if raw in o: cov.append(dict(source=s,attr=col,non_null=int(o[raw].notna().sum()),parsed=int(o[col].notna().sum())))
    rows[s]=o
allr=pd.concat(rows.values(),ignore_index=True)
allr.to_pickle('work/state/normalized.pkl'); allr.to_csv('work/state/normalized.csv',index=False)
print(pd.DataFrame(cov))
print(allr.groupby('source').apply(lambda g: g.notna().mean()).T)
