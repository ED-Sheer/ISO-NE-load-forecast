# ISO-NE 负荷预测：阶段 1 报告

## 0. TL;DR
- 数据: ISO-NE 2024 小时级负荷，8781 小时
- 任务: 预测未来 24 小时 RT_Demand
- 方法: LightGBM + 时间/lag/rolling 特征，滑动窗口回测
- 结果: 滚动 MAPE = 5.19%，打赢 naive 基线 (5.95%) 0.77 个百分点
- 阶段 1 范围: 单步预测、无天气/节假日特征

## 1. 数据
- 来源、时间范围、频率、缺失情况
- 图: 01_load_curve.png
- 关键观察: 夏季主峰 (7-8月)，冬季次峰 (12-1月)，
  日内峰谷差 6000-8000 MW

## 2. 时序切分
- 为什么不能随机切分
- 三种切分方式：固定 / 滑动 / 扩展
- 本报告采用：滑动窗口 180 天
- 图: 02_split_diagram.png
- 自检: train.max < test.min, 训练窗口恒为 4320h

## 3. 特征工程
- 时间特征: hour, dayofweek, is_weekend
- lag 特征: lag_24, lag_168
- rolling 特征: rolling_mean_24, rolling_mean_168 (shift 1)
- 防泄露原则: 所有 lag/rolling 都 shift，t 时刻只看 t 之前
- 图: 09_feature_importance.png
- 阶段 2 引入了 Open-Meteo 的逐小时气温，构造 HDD/CDD、温度滞后项，并引入美国法定节假日标记，使滚动回测 MAPE 从 5.19% 降至 3.55%。

## 4. 模型
- LightGBM, 500 棵树, lr=0.05, num_leaves=31
- 不调参，不早停 (原因见 D6 笔记)
- 每段重新训练

## 5. 评估指标
- MAE, RMSE, MAPE, MASE
- MASE 分母用训练集 seasonal naive (lag=24)
- 除零保护: y_true==0 跳过

## 6. 结果

### 6.1 主结果表 (滚动回测, mean)
| 方法                                    | MAPE      | MAE | RMSE | MASE |
|---------------------------------------|-----------|---|---|---|
| naive (lag=24)                        | 5.95%     | 812.7 | 958.2 | 0.906 |
| LightGBM time-only                    | 13.32%    | 1860.9 | 2030.7 | 2.086 |
| **LightGBM full**                     | **5.19%** | **722.3** | **859.4** | **0.808** |
| **LightGBM + Weather (D8)**           | **3.62%** |           |           |           |
| **LightGBM + Weather + Holiday (D8)** | **3.55%** |           |           |           |

图: 05_mape_comparison.png

### 6.2 逐月误差
图: 08_by_month.png
- 7月和10月改善最大 (lag_24 修季节漂移)
- 12月改善最小 (寒潮+假期)

### 6.3 按星期几
图: 07_by_dow.png
- 周一/周六改善大 (lag_168 修周内错位)
- 周三/周四轻微恶化 (+0.3)，平稳日过拟合

### 6.4 特征重要性
- rolling_mean_168 第一
- lag_24 / lag_168 并列
- hour 掉到第五

## 7. 已知限制
1. 8/30 劳动节前周五，naive MAPE 20.4%，lag 只能部分修
2. 夏季高温日残余尖峰 (7-8月 MAPE 8-10%)
3. 周三/周四轻微过拟合
4. 单步预测 (horizon=24)，多步会更难
5. 仅 1 年数据，365 天窗口无法测试

## 8. 阶段 2 方向
- 天气特征 (气温、湿度)
- 节假日编码
- 多步预测 (horizon=72)
- 概率预测 (区间)
- 至少 2 年数据