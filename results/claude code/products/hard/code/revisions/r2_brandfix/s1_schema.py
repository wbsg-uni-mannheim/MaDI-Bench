"""Stage 1: schema matching. Columns mapped by meaning; verified by value inspection
(column order is identical across sources and each column's values were checked)."""
import pandas as pd, json
TARGET = ['id','brand','title','description','price','priceCurrency','url','model','model_number',
          'product_type','chipset_name','vram_gb','storage_gb','read_speed_mb_s','write_speed_mb_s',
          'bus_type','interface_type','width_mm','length_mm','height_mm','weight_g',
          'storage_connection_type','memory_type','color','form_factor']
# explicit per-source mapping (source column -> target attribute), confidence
MAP = {
 'products_1': dict(zip(['id','brnd','prd_ttl','desc','prc','cur','product_url','mdl','mdl_no','prd_typ','chip','vram','stor','rd_speed_mbs','wr_speed_mbs','bus_t','iface_t','w_mm','l_mm','h_mm','wt_g','stor_conn','mem_t','clr','form_f'], TARGET)),
 'products_2': dict(zip(['id','br','t','dsc','amt','ccy','productUrl','m','mno','ptyp','cn','vrm','stg','rs','ws','bt','it','wd','ln','ht','wt','sc','mt','col','ff'], TARGET)),
 'products_3': dict(zip(['id','Attribute_3','Attribute_2','Attribute_4','Attribute_5','Attribute_6','Link','Attribute_9','Attribute_10','Attribute_8','Attribute_11','Attribute_12','Attribute_13','Attribute_19','Attribute_20','Attribute_14','Attribute_15','Attribute_22','Attribute_23','Attribute_24','Attribute_25','Attribute_17','Attribute_16','Attribute_21','Attribute_18'], TARGET)),
 'products_4': dict(zip(['id','br','t','dsc','amt','ccy','link','m','mno','ptyp','cn','vrm','stg','rs','ws','bt','it','wd','ln','ht','wt','sc','mt','col','ff'], TARGET)),
}
# lower confidence where semantics are ambiguous / units inconsistent in source
LOWCONF = {'storage_gb':0.9,'weight_g':0.8,'width_mm':0.8,'length_mm':0.8,'height_mm':0.8,'storage_connection_type':0.9,'form_factor':0.9}
rows=[]; frames=[]
for src, m in MAP.items():
    d = pd.read_csv(f'task/input/data/{src}.csv', dtype=str, keep_default_na=False)
    assert list(d.columns)==list(m.keys()), src
    d = d.rename(columns=m); d.insert(1,'source',src)
    frames.append(d)
    for s,t in m.items():
        rows.append(dict(source_dataset=src, source_column=s, target_dataset='hardware_product', target_column=t, score=LOWCONF.get(t,1.0)))
pd.DataFrame(rows).to_csv('submission/sm_mapping.csv', index=False)
a = pd.concat(frames, ignore_index=True)
a = a.apply(lambda c: c.str.strip() if c.dtype==object else c)
a.to_csv('work/state/s1_translated.csv', index=False)
print(a.shape)
