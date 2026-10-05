import pandas as pd, numpy as np, sys
f=pd.read_pickle('/tmp/fs.pkl'); u=pd.read_pickle('work/state/s2_normalized.pkl').set_index('id')
b=f.sort_values('score',ascending=False).drop_duplicates(['pair','id1'])
def show(r):
  print(round(r.score,1), 'ts=%.2f/%.2f pre=%d main=%d as=%.2f yd=%s ven=%s vol=%s fp=%s'%(r.tsim,r.tsort,r.prefix,r.main,r.asim,r.ydiff,r.venue,r.vol_eq,r.fp_eq))
  for i in (r.id1,r.id2):
    x=u.loc[i]; print('   ',i,'|',x.title_n[:90],'|',x.authors_l[:3],'|',x.year_n,'|',x.journal_n[:40],'|',x.volume_n,x.first_page_n,x.type_n)
for lo,hi in [(float(a),float(b_)) for a,b_ in zip(sys.argv[1::2],sys.argv[2::2])]:
  print('=== bin',lo,hi)
  x=b[(b.score>=lo)&(b.score<hi)]
  for _,r in x.sample(min(8,len(x)),random_state=3).iterrows(): show(r)
