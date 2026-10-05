"""Stage 1: schema matching. Manual semantic mapping (inspected types/values), writes
sm_mapping.csv and a unified raw table (source-native ids, raw values untouched)."""
import pandas as pd, os
D='task/input/data'
MAP={
 'dbpedia':{'id':'id','org_nm':'name','est_yr':'founded','ctry':'country','hq_city':'city',
            'sect':'industry','kp_nm':'keypeople','tot_ast':'assets','ann_inc':'revenue'},
 'forbes':{'id':'id','name':'name','country':'country','industry':'industry','assets':'assets','revenue':'revenue'},
 'fullcontact':{'id':'id','nm':'name','cn':'country','cy':'city','kp':'keypeople','fy':'founded'},
}
UNMAPPED={'forbes':{'website':'Forbes profile URL (duplicates id); no target attribute'}}
SCORE={('dbpedia','ann_inc'):0.8,('dbpedia','kp_nm'):0.8,('fullcontact','kp'):0.8,('dbpedia','sect'):0.9}
rows=[];frames=[]
TARGET=['id','name','founded','country','city','industry','assets','revenue','keypeople']
for src,m in MAP.items():
    df=pd.read_csv(f'{D}/{src}.csv',dtype=str,keep_default_na=False,na_values=[''])
    assert set(m)<=set(df.columns)
    for sc,tc in m.items(): rows.append((src,sc,'companies',tc,SCORE.get((src,sc),1.0)))
    out=pd.DataFrame({'source':src,'record_id':df['id']})
    for sc,tc in m.items():
        if tc!='id': out[tc+'_raw']=df[sc]
    for tc in TARGET[1:]:
        if tc+'_raw' not in out: out[tc+'_raw']=None
    frames.append(out)
os.makedirs('submission',exist_ok=True)
pd.DataFrame(rows,columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv('submission/sm_mapping.csv',index=False)
u=pd.concat(frames,ignore_index=True)
assert not u.record_id.duplicated().any()
u.to_pickle('work/state/s1_unified_raw.pkl')
print(u.groupby('source').count().T)
