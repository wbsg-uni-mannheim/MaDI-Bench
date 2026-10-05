"""Stage 1: schema matching. Sources share a 25-column layout with renamed headers;
mapping chosen by header semantics and verified against value examples (see report)."""
import pandas as pd, json, os
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
TARGET = ['id','brand','title','description','price','priceCurrency','url','model','model_number',
 'product_type','chipset_name','vram_gb','storage_gb','read_speed_mb_s','write_speed_mb_s','bus_type',
 'interface_type','width_mm','length_mm','height_mm','weight_g','storage_connection_type','memory_type','color','form_factor']
MAP = {
 'products_1': dict(zip(['id','brand','title','description','price','priceCurrency','product_url','model','model_number','product_type','chipset_name','vram_gb','storage_gb','read_speed_mb_s','write_speed_mb_s','bus_type','interface_type','width_mm','length_mm','height_mm','weight_g','storage_connection_type','memory_type','color','form_factor'], TARGET)),
 'products_2': dict(zip(['id','brnd','prd_ttl','desc','prc','cur','productUrl','mdl','mdl_no','prd_typ','chip','vram','stor','rd_speed_mbs','wr_speed_mbs','bus_t','iface_t','w_mm','l_mm','h_mm','wt_g','stor_conn','mem_t','clr','form_f'], TARGET)),
 'products_3': dict(zip(['id','br','t','dsc','amt','ccy','Link','m','mno','ptyp','cn','vrm','stg','rs','ws','bt','it','wd','ln','ht','wt','sc','mt','col','ff'], TARGET)),
 'products_4': dict(zip(['id','brnd','prd_ttl','desc','prc','cur','link','mdl','mdl_no','prd_typ','chip','vram','stor','rd_speed_mbs','wr_speed_mbs','bus_t','iface_t','w_mm','l_mm','h_mm','wt_g','stor_conn','mem_t','clr','form_f'], TARGET)),
}
def main():
    frames, rows = [], []
    for src, m in MAP.items():
        d = pd.read_csv(f'{ROOT}/task/input/data/{src}.csv', dtype=str, keep_default_na=False)
        assert set(d.columns) == set(m), (src, set(d.columns) ^ set(m))
        d = d.rename(columns=m)[TARGET]
        d.insert(0, 'source', src)
        frames.append(d)
        for s, t in m.items():
            rows.append(dict(source_dataset=src, source_column=s, target_dataset='target_schema', target_column=t, score=1.0))
    t = pd.concat(frames, ignore_index=True)
    t = t.apply(lambda c: c.str.strip())
    t.to_csv(f'{W}/state/translated.csv', index=False)
    os.makedirs(f'{ROOT}/submission', exist_ok=True)
    pd.DataFrame(rows).to_csv(f'{ROOT}/submission/sm_mapping.csv', index=False)
    print(t.shape, t.id.is_unique)

if __name__ == '__main__':
    main()
