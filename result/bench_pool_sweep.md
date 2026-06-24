| config | pool | docs/s | p50 ms | p90 ms | VRAM MiB | tables | formulas | infer tbl ms | infer fml ms |
|---|---|---|---|---|---|---|---|---|---|
| text_only@pool1 | 1 | 252.5 | 1.7 | 3.7 | 2466 | 0 | 0 |  |  |
| text_only@pool2 | 2 | 287.76 | 1.4 | 4.4 | 4276 | 0 | 0 |  |  |
| text_only@pool4 | 4 | 267.0 | 1.7 | 2.8 | 7896 | 0 | 0 |  |  |
| text_only@pool8 | 8 | 276.46 | 1.6 | 20.5 | 15142 | 0 | 0 |  |  |
| layout_only@pool1 | 1 | 142.94 | 28.8 | 53.4 | 3270 | 0 | 0 |  |  |
| layout_only@pool2 | 2 | 155.95 | 26.2 | 45.6 | 5824 | 0 | 0 |  |  |
| layout_only@pool4 | 4 | 142.73 | 25.4 | 67.6 | 10912 | 0 | 0 |  |  |
| layout_only@pool8 | 8 | 132.86 | 30.5 | 57.9 | 21082 | 0 | 0 |  |  |
| both_local@pool1 | 1 | 129.33 | 37.8 | 51.4 | 3548 | 3 | 11 | 144.5 |  |
| both_local@pool2 | 2 | 164.1 | 24.6 | 41.1 | 6346 | 3 | 11 | 153.4 |  |
| both_local@pool4 | 4 | 141.36 | 26.5 | 54.1 | 11926 | 3 | 11 | 165.4 |  |
| both_local@pool8 | 8 | 110.38 | 38.7 | 63.6 | 23084 | 3 | 0 | 10.7 |  |
