**Image set:** OmniDocBench 125-doc stratified subset via scripts/omnidoc_subset_n.py (identical docs to the accuracy harness): 125 images; ground truth = 119 tables, 489 formulas, 1294 text blocks  
**Concurrency:** 8

| config | pool | images/s | p50 ms | p90 ms | VRAM MiB | tables (/119 GT) | formulas (/489 GT) | infer tbl ms | infer fml ms |
|---|---|---|---|---|---|---|---|---|---|
| text_only@pool2 | 2 | 324.18 | 1.3 | 2.9 | 4273 | 0 | 0 |  |  |
| layout_only@pool2 | 2 | 56.07 | 49.9 | 124.7 | 5821 | 0 | 0 |  |  |
| table_slanext@pool2 | 2 | 86.27 | 56.6 | 123.4 | 5967 | 127 | 0 | 146.3 |  |
| table_vl@pool2 | 2 | 10.27 | 278.2 | 1571.6 | 5823 | 127 | 0 | 559.3 |  |
| formula_local@pool2 | 2 | 92.36 | 48.5 | 99.2 | 6197 | 0 | 0 |  |  |
| formula_vl@pool2 | 2 | 6.74 | 613.1 | 3269.8 | 5823 | 0 | 1679 |  | 900.2 |
| both_local@pool2 | 2 | 55.38 | 83.0 | 144.6 | 6343 | 127 | 0 | 148.6 |  |
| both_vl@pool2 | 2 | 6.1 | 721.3 | 1924.2 | 5823 | 127 | 1679 | 568.0 | 897.8 |
| hybrid_slanext_vlformula@pool2 | 2 | 12.1 | 343.7 | 1799.2 | 5969 | 127 | 1679 | 145.9 | 898.6 |
