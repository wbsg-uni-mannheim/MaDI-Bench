"""Stage 1: schema mapping (manual, semantics-based; see report)."""
import csv, json
M = [
 # source, column, target, score, transformation
 ("dbpedia","title","name",1.0,"strip wikipedia disambiguation '(… video game)'"),
 ("dbpedia","launch_yr","releaseYear",1.0,"ISO date; only year reliable"),
 ("dbpedia","studio","developer",0.9,"wiki entity names; noisy cross-product"),
 ("dbpedia","system","platform",1.0,"map to canonical platform vocabulary"),
 ("dbpedia","genre","genres",0.9,"wiki genre article names -> list"),
 ("dbpedia","franchise","series",1.0,"strip wiki disambiguation"),
 ("metacritic","game_title","name",1.0,""),
 ("metacritic","year_published","releaseYear",1.0,""),
 ("metacritic","made_by","developer",1.0,""),
 ("metacritic","console","platform",1.0,""),
 ("metacritic","genres","genres",1.0,"comma list, dedup"),
 ("metacritic","press_rating","criticScore",1.0,"float->int"),
 ("metacritic","player_rating","userScore",1.0,""),
 ("metacritic","age_rating","ESRB",1.0,"alias K-A->E"),
 ("sales","prod_title","name",1.0,""),
 ("sales","launch_dt","releaseYear",1.0,""),
 ("sales","studio","developer",1.0,""),
 ("sales","dist","publisher",1.0,""),
 ("sales","hw","platform",1.0,""),
 ("sales","genre","genres",1.0,"single value -> list"),
 ("sales","press_score","criticScore",1.0,""),
 ("sales","comm_rating","userScore",1.0,""),
 ("sales","age_classification","ESRB",1.0,"alias K-A->E"),
]
# unmapped: ids (provenance only; target 'id' is the fused _id), sales.units_sold_mm (no target attribute)
with open("submission/sm_mapping.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["source_dataset","source_column","target_dataset","target_column","score"])
    for s,c,t,sc,_ in M: w.writerow([s,c,"target",t,sc])
json.dump([dict(source=s,column=c,target=t,score=sc,transform=x) for s,c,t,sc,x in M]+
          [dict(source="sales",column="units_sold_mm",target=None,note="no target attribute"),
           dict(source="*",column="id cols",target=None,note="kept as record_id for provenance")],
          open("work/state/schema_mapping.json","w"),indent=1)
print("wrote", len(M))
