"""Stage 1: schema matching. All 4 sources expose the same 25 semantic fields under
different names; mapping decided from metadata descriptions + value inspection."""
import json, pandas as pd
CANON = ['id','brand','title','description','price','priceCurrency','url','model','model_number',
         'product_type','chipset_name','vram_gb','storage_gb','read_speed_mb_s','write_speed_mb_s',
         'bus_type','interface_type','width_mm','length_mm','height_mm','weight_g',
         'storage_connection_type','memory_type','color','form_factor']
rows, tabs = [], []
for i in range(1, 5):
    name = f'dataset_{i}'
    meta = json.load(open(f'../task/input/data/{name}_metadata.json'))
    cols = [v['name'] for v in meta['variableMeasured']]
    data = json.load(open(f'../task/input/data/{name}.json'))
    assert list(data[0].keys()) == cols and len(cols) == len(CANON)
    for sc, tc in zip(cols, CANON):
        # length/depth columns ("depthMm") mapped to length_mm: metadata says "Length or depth"
        score = 0.9 if sc in ('depthMm',) else 1.0
        rows.append(dict(source_dataset=name, source_column=sc, target_dataset='target_schema',
                         target_column=tc, score=score))
    df = pd.DataFrame(data)[cols]
    df.columns = CANON
    df.insert(0, 'source', name)
    df['id'] = df['id'].astype(str)
    tabs.append(df)
pd.DataFrame(rows).to_csv('state/sm_mapping.csv', index=False)
all_ = pd.concat(tabs, ignore_index=True)
all_.to_pickle('state/s1_translated.pkl')
print(all_.shape, all_.groupby('source').size().to_dict())
