"""Stage 1: schema mapping (rule-based, by inspected meaning/units)."""
import json, pandas as pd, os
W=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(W)
M=[ # source, column, target, score, transformation
 ("dbpedia","id","id",1.0,"native id kept verbatim"),
 ("dbpedia","name","name",1.0,"strip '(disambiguation)' + legal suffix"),
 ("dbpedia","founded","founded",1.0,"'January 01, YYYY' -> YYYY-01-01"),
 ("dbpedia","country","country",1.0,"CLDR alias mapping"),
 ("dbpedia","city","city",0.9,"trimmed; noisy concatenations kept"),
 ("dbpedia","industry","industry",0.8,"free-text DBpedia industry -> GICS Industry Name via alias table"),
 ("dbpedia","keypeople","keypeople",0.9,"founders list; parse python-list strings"),
 ("dbpedia","assets","assets",0.7,"mixed units: <100 -> bn, <1e5 -> mn, else raw US$"),
 ("dbpedia","revenue","revenue",0.7,"mixed units: same heuristic as assets"),
 ("forbes","id","id",1.0,"native id kept verbatim"),
 ("forbes","name","name",1.0,"legal suffix stripped"),
 ("forbes","country","country",1.0,"CLDR alias mapping"),
 ("forbes","industry","industry",0.9,"Forbes industry -> GICS Industry Name via alias table"),
 ("forbes","assets","assets",1.0,"US$ billions *1e9"),
 ("forbes","revenue","revenue",1.0,"US$ billions *1e9"),
 ("fullcontact","id","id",1.0,"native id kept verbatim"),
 ("fullcontact","name","name",1.0,"legal suffix stripped"),
 ("fullcontact","country","country",1.0,"CLDR alias mapping"),
 ("fullcontact","city","city",1.0,"trimmed"),
 ("fullcontact","keypeople","keypeople",0.9,"founders list"),
 ("fullcontact","founded","founded",1.0,"ISO or 'January 01, YYYY' -> YYYY-01-01"),
]
schema=json.load(open(f"{ROOT}/task/input/schemamatching/target_schema.json"))
for s,c,t,_,_ in M:
    assert c in pd.read_csv(f"{ROOT}/task/input/data/{s}.csv",nrows=1).columns, (s,c)
    assert t in schema["properties"], t
df=pd.DataFrame(M,columns=["source_dataset","source_column","target_column","score","transformation"])
df["target_dataset"]="companies"
os.makedirs(f"{ROOT}/submission",exist_ok=True)
df[["source_dataset","source_column","target_dataset","target_column","score"]].to_csv(f"{ROOT}/submission/sm_mapping.csv",index=False)
df.to_csv(f"{W}/state/schema_mapping_inventory.csv",index=False)
# unmapped: forbes.website (duplicate of id URL, used only as matching evidence via slug)
print(df.groupby("source_dataset").size())
