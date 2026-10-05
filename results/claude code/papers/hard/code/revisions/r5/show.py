import pandas as pd, sys
u = pd.read_pickle('work/state/s1_translated.pkl').set_index('id')
C = ['type','title','authors','publication_year','journal','volume','issue','first_page','last_page','referenced_works_count','cited_by_count']
def show(ids):
    for i in ids:
        r = u.loc[i]
        print(i, ' | '.join(f'{c[:4]}={str(r[c])[:90]}' for c in C if r[c] != ''))
if __name__ == '__main__':
    show(sys.argv[1:])
