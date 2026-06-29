# Fusion Ground Truth

This directory holds the hand-annotated data-fusion validation and test records for this task.

For a first look, prefer the `*_better_readability.csv` files. They keep the `source_ids` helper column followed by the manually validated hardware fact columns in a compact order. `source_ids` lists all released product source records that describe the fused product, for example `products_1_...`, `products_2_...`, `products_3_...`, and `products_4_...`.

`price`, `priceCurrency`, and `title` are intentionally excluded from the fusion ground truth. Vendor offer prices are time- and source-dependent, and product titles differ by source listing, so neither forms a stable product-level truth. Source descriptions, URLs, raw left/right values, sampling labels, cluster IDs, and ground-truth provenance URLs are also excluded from the fusion target surface.

The fusion files keep only helper IDs and manually validated product facts. Where a target-schema field is missing from an original split, the Better Readability CSV leaves that column empty rather than inferring a value.
