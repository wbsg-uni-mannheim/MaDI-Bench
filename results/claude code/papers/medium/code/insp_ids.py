import pandas as pd, sys
u=pd.read_pickle('work/state/s2_normalized.pkl').set_index('id')
def show(i):
    x=u.loc[i]; print('   ',i,'|',x.title_n[:80],'|',x.authors_l[:3],'|',x.year_n,'|',x.journal_n[:35],'|',x.volume_n,x.issue_n,x.first_page_n,x.last_page_n,x.type_n,'|',x.cite_n)
