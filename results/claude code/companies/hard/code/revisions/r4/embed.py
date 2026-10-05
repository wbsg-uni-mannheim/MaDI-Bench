"""Embedding helper with on-disk cache (rebuild uses cache only; no network if all texts cached)."""
import os, pickle, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f'{ROOT}/work/state/emb_cache.pkl'
MODEL = 'text-embedding-3-small'
_c = None
def _load():
    global _c
    if _c is None:
        _c = pickle.load(open(CACHE,'rb')) if os.path.exists(CACHE) else {}
    return _c
def embed(texts, model=MODEL):
    c = _load()
    missing = sorted({t for t in texts if (model, t) not in c})
    if missing:
        if os.environ.get('NO_NETWORK'): raise RuntimeError(f'{len(missing)} texts not in embedding cache')
        import openai
        cl = openai.OpenAI()
        for i in range(0, len(missing), 500):
            batch = missing[i:i+500]
            r = cl.embeddings.create(model=model, input=[b if b else ' ' for b in batch])
            for t, d in zip(batch, r.data): c[(model, t)] = np.array(d.embedding, dtype=np.float32)
        pickle.dump(c, open(CACHE,'wb'))
    X = np.stack([c[(model, t)] for t in texts])
    return X / np.linalg.norm(X, axis=1, keepdims=True)
