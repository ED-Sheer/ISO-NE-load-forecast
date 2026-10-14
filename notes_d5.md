## D5 特征重要性 (单次切分 full 模型)
| 特征 | Split Count |
|---|---|
| rolling_mean_168 | 4673 |
| rolling_mean_24  | 3637 |
| lag_24           | 2409 |
| lag_168          | 2405 |
| hour             | 961  |
| dayofweek        | 704  |
| is_weekend       | 211  |

## 解读
- rolling_mean_168 第一: 滚动回测下，模型靠它推断"当前季节水平"
- lag_24 vs lag_168 并列: 信息高度重叠，树在两者间分摊
- hour 掉到第五: lag/rolling 吃掉了日内周期信息，模型优先用最近实际值
- is_weekend 垫底: 和 dayofweek 信息重叠，且 lag 已隐含周末信息

## 对 D6 的启示
- 不砍特征，保持现状
- 周三周四的轻微过拟合与特征无关，是模型复杂度问题，用 early stopping 治
- rolling_mean_168 的强势说明"当前水平"是关键，阶段 2 的天气特征会和它配合