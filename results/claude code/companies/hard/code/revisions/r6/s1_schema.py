"""Stage 1: schema mapping (manual, evidence-based; see report)."""
import csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 'companies'
MAP = [
 # source, column, target, score, transformation / reason
 ('dbpedia','id','id',1.0,'native id (DBpedia resource URI); URI slug also used as clean-name evidence'),
 ('dbpedia','nm','name',1.0,'company name; legal suffix + disambiguator stripped'),
 ('dbpedia','ey','founded',0.95,'establishment date, mixed formats -> YYYY-01-01 style ISO'),
 ('dbpedia','cn','country',0.95,'country, noisy -> CLDR name'),
 ('dbpedia','hq','city',0.8,'headquarters location string, concatenated fields -> first city token'),
 ('dbpedia','sg','industry',0.8,'industry/segment free text -> GICS industry name'),
 ('dbpedia','kpn','keypeople',0.7,'key people free text -> list of person names (roles stripped)'),
 ('dbpedia','ta','assets',0.9,'total assets, mixed locale/unit formats -> USD integer'),
 ('dbpedia','ai','revenue',0.75,'annual income/revenue; magnitudes agree with Forbes sales -> revenue'),
 ('forbes','id','id',1.0,'native id (Forbes company URL)'),
 ('forbes','co_nm','name',1.0,'company name'),
 ('forbes','region','country',0.95,'country, noisy -> CLDR name'),
 ('forbes','bus_seg','industry',0.9,'Forbes industry -> GICS industry name'),
 ('forbes','ast_val','assets',1.0,'assets, mixed formats/scales -> USD integer'),
 ('forbes','sls_fig','revenue',1.0,'sales -> revenue USD integer'),
 ('fullcontact','id','id',1.0,'native id'),
 ('fullcontact','Attribute_2','name',0.95,'organisation name'),
 ('fullcontact','Attribute_3','country',0.95,'country'),
 ('fullcontact','Attribute_4','city',0.9,'city/locality'),
 ('fullcontact','Attribute_5','keypeople',0.7,'people free text -> list of names'),
 ('fullcontact','Attribute_6','founded',0.9,'founding date'),
]
# forbes.url is a within-source link (duplicate rows point to their canonical page); not a target attribute.
if __name__ == '__main__':
    os.makedirs(f'{ROOT}/submission', exist_ok=True)
    with open(f'{ROOT}/submission/sm_mapping.csv','w',newline='') as fh:
        w = csv.writer(fh); w.writerow(['source_dataset','source_column','target_dataset','target_column','score'])
        for s,c,t,sc,_ in MAP: w.writerow([s,c,T,t,sc])
    with open(f'{ROOT}/work/state/schema_mapping_notes.csv','w',newline='') as fh:
        w = csv.writer(fh); w.writerow(['source','column','target','score','transformation'])
        for r in MAP: w.writerow(r)
    print('wrote', len(MAP))
