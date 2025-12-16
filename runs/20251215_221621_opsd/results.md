# Run Results

- **Dataset:** `opsd`
- **Config:** `configs/opsd_lstm.yaml`
- **Checkpoint:** `checkpoints/opsd_best.pt`
- **Selection:** metric=`RMSE`, mode=`min`
- **Best epoch:** `2`

**Key takeaway:** The residual LSTM beats a strong seasonal-naive baseline (lag-168) on the test set.
- **Test RMSE improvement vs seasonal naive (lag-168):** `12.87%` (lower is better)

## Validation

| Model | RMSE | MAE | sMAPE |
|---|---:|---:|---:|
| **Neural net** | 3813.74 | 2321.59 | 4.29 |
| Baseline `lag24` | 7118.77 | 4665.41 | 8.61 |
| Baseline `lag168` | 3992.64 | 2295.22 | 4.21 |

## Test

| Model | RMSE | MAE | sMAPE |
|---|---:|---:|---:|
| **Neural net** | 3627.84 | 2347.32 | 4.56 |
| Baseline `lag24` | 6182.09 | 4085.16 | 7.80 |
| Baseline `lag168` | 4163.58 | 2436.98 | 4.63 |
