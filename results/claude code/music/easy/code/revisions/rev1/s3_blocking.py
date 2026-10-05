import pandas as pd, numpy as np, json, sys, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,'work')
a=pd.read_pickle('work/state/normalized.pkl')
a['akey']=a.artist_keys.map(lambda s:' '.join(json.loads(s)))
a['na']=a.name_key+' '+a.akey
a['tk']=a.tracks_key.map(lambda s:' | '.join(json.loads(s)))
a['ntr']=a.tracks_key.map(lambda s:len(json.loads(s)))
SRC=['discogs','lastfm','musicbrainz']
def topk(field, analyzer, ngram, k, thr, mask=None, name=''):
    v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=1,sublinear_tf=True)
    X=v.fit_transform(a[field].fillna(''))
    out=[]
    for s1,s2 in itertools.combinations(SRC,2):
        i1=np.where((a.source==s1)&(mask if mask is not None else True))[0]
        i2=np.where((a.source==s2)&(mask if mask is not None else True))[0]
        for q,t in [(i1,i2),(i2,i1)]:
            for st in range(0,len(q),2000):
                qq=q[st:st+2000]
                S=(X[qq]@X[t].T).toarray()
                kk=min(k,S.shape[1])
                idx=np.argpartition(-S,kk-1,axis=1)[:,:kk]
                for r in range(len(qq)):
                    for c in idx[r]:
                        if S[r,c]>=thr: out.append((qq[r],t[c]))
    print(name,len(out))
    return out
P=set()
methods={}
for nm,args in {'name_char':('name_key','char_wb',(3,3),15,0.35,None),
                'nameartist_char':('na','char_wb',(3,3),15,0.35,None),
                'tracks_word':('tk','word',(1,2),5,0.4,(a.ntr>=2).values)}.items():
    ps=topk(*args,name=nm)
    ps={tuple(sorted(p)) for p in ps}
    methods[nm]=ps; P|=ps
ids=a.id.values
c=pd.DataFrame(sorted(P),columns=['i1','i2'])
c['id1']=ids[c.i1];c['id2']=ids[c.i2]
for nm,ps in methods.items(): c[nm]=[p in ps for p in zip(c.i1,c.i2)]
c.to_pickle('work/state/candidates.pkl')
c[['id1','id2']].to_csv('submission/blocking/candidates.csv',index=False)
# diagnostics
c['pair']=a.source.values[c.i1]+'-'+a.source.values[c.i2]
diag={'n_candidates':len(c),'by_pair':c.pair.value_counts().to_dict(),
      'unique_contrib':{m:int((c[m]&~c[[x for x in methods if x!=m]].any(axis=1)).sum()) for m in methods}}
cnt=pd.Series(np.concatenate([c.i1,c.i2])).value_counts().reindex(range(len(a)),fill_value=0)
diag['zero_candidate_by_source']=a.assign(n=cnt.values).groupby('source').n.apply(lambda s:int((s==0).sum())).to_dict()
diag['max_per_record']=int(cnt.max())
tot=sum(len(a[a.source==x])*len(a[a.source==y]) for x,y in itertools.combinations(SRC,2))
diag['reduction_ratio']=1-len(c)/tot
print(json.dumps(diag,indent=1))
json.dump(diag,open('work/state/blocking_diag.json','w'),indent=1)
