| Method | Overall CRPS | Cloudy CRPS | Ramp CRPS | RMSE | MAE | 90% Coverage | Ramp Coverage | Sunrise Coverage |
|---|---|---|---|---|---|---|---|---|
| Original G3''-F | *0.573* | *0.894* | **2.411** | *1.914* | *0.775* | 0.901 | 0.651 | **0.604** |
| + Physics | **0.572** | **0.893** | *2.414* | **1.913** | **0.773** | *0.901* | 0.650 | *0.602* |
| + Ramp | 0.575 | 0.899 | 2.419 | 1.918 | 0.775 | 0.901 | *0.655* | 0.598 |
| + Physics + Ramp | 0.575 | 0.900 | 2.421 | 1.919 | 0.775 | 0.901 | 0.653 | 0.601 |
| + Physics + Ramp + Bounds | 0.575 | 0.900 | 2.425 | 1.919 | 0.775 | 0.901 | 0.654 | 0.600 |
| + Physics + Ramp + Bounds + Direction | 0.575 | 0.899 | 2.420 | 1.918 | 0.775 | **0.900** | **0.658** | 0.599 |

Bold = best, italic = second best (for coverage: closest to the 90% nominal level). Development out-of-fold G3''-F; CRPS, RMSE, MAE in kW.