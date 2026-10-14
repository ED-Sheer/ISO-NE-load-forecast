# build_features.py
import pandas as pd


# 全局定义哪些列是类别特征，后面训练时直接引用
CATEGORICAL_FEATURES = ['hour', 'dayofweek']


def build_features(df, feature_set = 'time_only'):
    """
    构造特征。D3 阶段只加时间特征。

    契约（D4 的 backtest 会按这个契约调用）：
    - 输入 df 必须有 DatetimeIndex。
    - 返回 DataFrame，索引与 df 完全一致（不 resample、不 reset_index）。
    - 只返回特征列，不包含 target。target 由调用方单独取。

    :param df: DataFrame，必须有 DatetimeIndex 和 RT_Demand 列
    :param feature_set:  - 'time_only' : hour / dayofweek / is_weekend
          - 'time_lag_rolling' : 上面的 + lag_24 / lag_168 / rolling_24 / rolling_168
    :return: DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_features 要求输入 df 的索引是 DatetimeIndex")

    feat = pd.DataFrame(index=df.index)
    feat['hour'] = df.index.hour
    feat['dayofweek'] = df.index.dayofweek
    feat['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    if feature_set in ('time_lag_rolling', 'all'):
        if 'RT_Demand' not in df.columns:
            raise ValueError("feature_set='time_lag_rolling' 需要 df 含 'RT_Demand' 列")
        s = df['RT_Demand']

        # lag：直接用 shift，t 时刻取 t-24 / t-168 的真实值
        feat['lag_24'] = s.shift(24)
        feat['lag_168'] = s.shift(168)

        # rolling：先算 rolling，再 shift(1)，保证不含当前点
        feat['rolling_mean_24'] = s.rolling(24).mean().shift(1)
        feat['rolling_mean_168'] = s.rolling(168).mean().shift(1)
    
    return feat
