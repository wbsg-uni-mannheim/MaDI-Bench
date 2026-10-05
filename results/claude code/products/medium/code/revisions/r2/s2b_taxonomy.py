"""Taxonomy plan + coverage measurement (label-free). Writes work/taxonomy_plan.json and work/taxonomy_coverage.csv."""
import pandas as pd, json, os, re
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W); T = f'{ROOT}/task/input/schemamatching'
n = pd.read_pickle(f'{W}/state/normalized.pkl')
gm = pd.read_csv(f'{T}/GPU_Memory_Taxonomy.csv'); si = pd.read_csv(f'{T}/Storage_Interface_Taxonomy.csv'); pt = pd.read_csv(f'{T}/Product_Type_Taxonomy.csv')
plan = {
 'product_type': {'source': 'target_schema.json enum', 'values': ['GPU', 'SSD', 'HDD', 'USB_STICK'], 'exhaustive': True,
   'aliases': 'OCR/spelling variants and descriptive names mapped in s2_normalize.PT_ALIASES (e.g. "Solid State Drive"->SSD, "HXD"->HDD)',
   'unmapped_policy': 'records of other categories (monitors, RAM, ...) keep their own category (most common source spelling) - they are out of the schema enum and reported as violations',
   'Product_Type_Taxonomy.csv': 'subtypes (e.g. "External SSD", "M.2 SSD") roll up to the SSD/HDD enum; RAM/PSU/cooling subcategories have no enum value'},
 'memory_type': {'source': 'GPU_Memory_Taxonomy.csv', 'canonical_column': 'Variant', 'exhaustive': False,
   'policy': 'used for matching (GDDR5 vs GDDR6 conflict); fused value = majority source spelling (case-insensitive vote), not rewritten'},
 'bus_type/interface_type': {'source': 'Storage_Interface_Taxonomy.csv + schema expectedFamilies', 'exhaustive': False,
   'policy': 'used for matching (SATA/SAS/NVMe/USB families); fused value = majority source spelling; no invented generations'},
 'list_serialization': 'no array-typed attributes in the target schema'}
json.dump(plan, open(f'{W}/taxonomy_plan.json', 'w'), indent=1)
gm_vals = {v.lower() for v in gm.Variant}
fam = ['pcie', 'pci express', 'pci-e', 'sata', 'sas', 'usb', 'thunderbolt', 'firewire', 'nvme', 'lightning', 'micro usb']
rows = []
for src, g in n.groupby('source'):
    v = g.n_product_type_raw.dropna()
    rows.append((src, 'product_type', len(v), int(v.isin(['GPU', 'SSD', 'HDD', 'USB_STICK']).sum())))
    v = g.memory_type[g.memory_type != '']
    rows.append((src, 'memory_type(GPU taxonomy variant)', len(v), int(v.str.lower().str.replace(' ', '').isin(gm_vals).sum())))
    for c in ['bus_type', 'interface_type']:
        v = g[c][g[c] != '']
        rows.append((src, c + '(family)', len(v), int(v.str.lower().apply(lambda x: any(f in x for f in fam)).sum())))
cov = pd.DataFrame(rows, columns=['source', 'attribute', 'non_null', 'canonical'])
cov['unmapped'] = cov.non_null - cov.canonical; cov['canonical_rate'] = (cov.canonical / cov.non_null).round(4)
cov.to_csv(f'{W}/taxonomy_coverage.csv', index=False)
print(cov.to_string())
