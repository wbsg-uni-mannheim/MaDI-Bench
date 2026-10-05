"""One-off (NOT run by rebuild.sh): embed distinct industry strings and GICS labels via the
permitted embeddings endpoint; cached to work/state/emb_cache.json for offline reuse."""
import os,json,requests,pandas as pd
U=pd.read_pickle('work/state/s1_unified_raw.pkl')
g=pd.read_csv('task/input/schemamatching/GICS_Industry_Taxonomy.csv',dtype=str)
texts=set(U.industry_raw.dropna().str.strip())|set(g['Industry Name'])|set(g['Sub-Industry Name'])
path='work/state/emb_cache.json'
cache=json.load(open(path)) if os.path.exists(path) else {}
todo=sorted(t for t in texts if t not in cache)
url=os.environ['OPENAI_BASE_URL'].rstrip('/')+'/embeddings'
for i in range(0,len(todo),200):
    b=todo[i:i+200]
    r=requests.post(url,headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},json={'input':b,'model':'text-embedding-3-small'},timeout=120)
    r.raise_for_status()
    for t,e in zip(b,r.json()['data']): cache[t]=e['embedding']
json.dump(cache,open(path,'w'))
print(len(cache))
