# Results (latest run)

**Dataset:** `opsd`  
**Config:** `configs/opsd_lstm.yaml`  
**Run folder:** `runs/20251215_221621_opsd`

![Test RMSE comparison](runs/20251215_221621_opsd/report_rmse_test.png)

## Metrics table

(Generated from `runs/20251215_221621_opsd/results.yaml`)

| model           |   RMSE (test) |   RMSE (val) |   MAE (test) |   MAE (val) |   SMAPE (test) |   SMAPE (val) |
|:----------------|--------------:|-------------:|-------------:|------------:|---------------:|--------------:|
| Baseline lag168 |       4163.58 |      3992.64 |      2436.98 |     2295.22 |           4.63 |          4.21 |
| Baseline lag24  |       6182.09 |      7118.77 |      4085.16 |     4665.41 |           7.8  |          8.61 |
| Neural net      |       3627.84 |      3813.74 |      2347.32 |     2321.59 |           4.56 |          4.29 |

Full table file: `runs/20251215_221621_opsd/report_table.md`
