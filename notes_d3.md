# D3 LightGBM (只加时间特征)

- 切分点: 2024-07-01
- 特征: ['hour', 'dayofweek', 'is_weekend']
- 参数: {'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1, 'n_estimators': 500, 'verbose': -1, 'random_state': 42}

| 指标 | naive 基线 | LightGBM | 变化 |
|---|---|---|---|
| MAE  | 802.2200  | 1745.0010  | +942.7810 |
| RMSE | 1130.2500 | 2394.6517 | +1264.4017 |
| MAPE | 5.8888% | 12.1510% | +6.2622 |
| MASE | 0.9702 | 2.1104 | +1.1402 |
