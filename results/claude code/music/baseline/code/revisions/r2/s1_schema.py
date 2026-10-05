"""Stage 1: schema mapping (manual, by semantics + example inspection)."""
import csv, json
MAP = [
 # source, column, target, score, transformation/note
 ("discogs","rec_uid","id",1.0,"native id kept"),
 ("discogs","title_str","name",1.0,"clean title"),
 ("discogs","performer","artist",1.0,""),
 ("discogs","pub_dt","release-date",1.0,"YYYY or YYYY-MM-DD"),
 ("discogs","origin_loc","release-country",1.0,"short names (UK) -> full country names"),
 ("discogs","duration","duration",1.0,"seconds; 0 = missing"),
 ("discogs","imprint","label",1.0,"'|'-separated list"),
 ("discogs","category","genre",0.8,"Discogs genre list -> taxonomy Genre Name (first mappable)"),
 ("discogs","tracks_track-name","tracks",1.0,"python-literal list"),
 ("lastfm","item_code","id",1.0,"native id kept"),
 ("lastfm","album_title","name",0.9,"noisy: prefixes/artist prefix stripped"),
 ("lastfm","band","artist",0.9,"noisy: first names abbreviated"),
 ("lastfm","album_length","duration",1.0,"seconds"),
 ("lastfm","tracks_track-name","tracks",1.0,"python-literal list"),
 ("musicbrainz","Attribute_1","id",1.0,"native id kept"),
 ("musicbrainz","Attribute_2","name",1.0,"suffix (orig.)/(album)/(release) stripped"),
 ("musicbrainz","Attribute_3","artist",1.0,"'Last, First' inverted form"),
 ("musicbrainz","Attribute_4","release-date",1.0,"YYYY-MM-DD or YYYY-MM"),
 ("musicbrainz","Attribute_5","release-country",1.0,"full country names"),
 ("musicbrainz","Attribute_6","duration",1.0,"seconds; 0 = missing"),
 ("musicbrainz","Attribute_9","tracks",1.0,"python-literal list"),
]
with open("submission/sm_mapping.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["source_dataset","source_column","target_dataset","target_column","score"])
    for s,c,t,sc,_ in MAP: w.writerow([s,c,"music_release",t,sc])
json.dump([dict(source=s,column=c,target=t,score=sc,note=n) for s,c,t,sc,n in MAP],open("work/state/mapping.json","w"),indent=1)
# validation
import pandas as pd
tgt=set(json.load(open("task/input/schemamatching/target_schema.json"))["properties"])
for s,c,t,_,_ in MAP:
    assert t in tgt, t
    assert c in pd.read_csv(f"task/input/data/{s}.csv",nrows=1).columns, (s,c)
print("mapping ok", len(MAP))
