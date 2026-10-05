"""Write submission files from the saved final-stage artifacts."""
import pandas as pd
m=pd.read_pickle('work/state/s5_membership.pkl'); F=pd.read_pickle('work/state/s6_fused.pkl'); C=pd.read_pickle('work/state/s5_correspondences.pkl')
m[['record_id','source','cluster_id']].sort_values(['cluster_id','record_id']).to_csv('submission/membership.csv',index=False)
F.sort_values('_id').to_csv('submission/fused.csv',index=False)
C.sort_values(['id1','id2']).to_csv('submission/correspondences.csv',index=False)
print('exported',len(m),len(F),len(C))
