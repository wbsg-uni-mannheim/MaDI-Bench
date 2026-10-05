"""Stage 1: schema mapping (rule-based, by meaning/type/examples)."""
import csv, json, os
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
MAP = [  # source, column, target, score, transformation
 ("dbpedia","id","id",1.0,"native id kept"),
 ("dbpedia","name","name",1.0,"title; wiki 'List of ...' pages kept as-is"),
 ("dbpedia","releaseYear","releaseYear",0.9,"'Month DD, YYYY' -> YYYY-MM-DD"),
 ("dbpedia","developer","developer",0.9,"wiki disambiguation suffix removed; platform-name contamination dropped"),
 ("dbpedia","platform","platform",1.0,"alias map to taxonomy"),
 ("dbpedia","genres","genres",0.9,"single genre per row, union per group"),
 ("dbpedia","series","series",1.0,"wiki disambiguation suffix removed"),
 ("metacritic","id","id",1.0,"native id kept"),
 ("metacritic","name","name",1.0,""),
 ("metacritic","releaseYear","releaseYear",1.0,"ISO date w/ whitespace noise"),
 ("metacritic","developer","developer",1.0,"comma list; first kept for fusion"),
 ("metacritic","platform","platform",1.0,"alias map"),
 ("metacritic","genres","genres",1.0,"comma list"),
 ("metacritic","criticScore","criticScore",1.0,"float->int"),
 ("metacritic","userScore","userScore",1.0,"0-10"),
 ("metacritic","ESRB","ESRB",1.0,"K-A->E"),
 ("sales","id","id",1.0,"native id kept"),
 ("sales","name","name",1.0,""),
 ("sales","releaseYear","releaseYear",0.9,"'Month DD, YYYY'"),
 ("sales","developer","developer",1.0,"some rows token-scrambled"),
 ("sales","publisher","publisher",1.0,"only source of publisher"),
 ("sales","platform","platform",1.0,"alias map"),
 ("sales","genres","genres",0.9,"some rows token-scrambled"),
 ("sales","criticScore","criticScore",1.0,"0-100"),
 ("sales","userScore","userScore",1.0,"0-10"),
 ("sales","ESRB","ESRB",1.0,"K-A->E"),
]
# unmapped: sales.globalSales (no target attribute)
os.makedirs(f"{ROOT}/submission", exist_ok=True)
with open(f"{ROOT}/submission/sm_mapping.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["source_dataset","source_column","target_dataset","target_column","score"])
    for s,c,t,sc,_ in MAP: w.writerow([s,c,"target",t,sc])
json.dump([dict(source=s,column=c,target=t,score=sc,transform=x) for s,c,t,sc,x in MAP]+
          [dict(source="sales",column="globalSales",target=None,score=0,transform="no target attribute; unmapped")],
          open(f"{W}/state/schema_mapping.json","w"),indent=1)
print("mapping rows",len(MAP))
