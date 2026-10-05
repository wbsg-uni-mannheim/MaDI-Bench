"""One-off: cache general-purpose name embeddings (inference only). Not called by rebuild.sh."""
import pandas as pd, numpy as np
from openai import OpenAI
N=pd.read_csv("state/normalized.csv")
texts=(N.name_raw.astype(str)).tolist()
c=OpenAI(); out=[]
for i in range(0,len(texts),1000):
    r=c.embeddings.create(model="text-embedding-3-small",input=texts[i:i+1000]); out+= [d.embedding for d in r.data]
E=np.array(out,dtype=np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)
np.save("state/emb_name.npy",E); N[["record_id"]].to_csv("state/emb_ids.csv",index=False); print(E.shape)
