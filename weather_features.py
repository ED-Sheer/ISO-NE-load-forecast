# weather_features.py
import numpy as np
import pandas as pd


def build_weather_features(df_index, temp_series, base_temp=18.3):
    """
    输入:
        df_index    : 负荷数据的 DatetimeIndex
        temp_series : 小时级气温 Series
    返回:
        DataFrame, index 与 df_index 一致
    """
    # 1. 强制对齐到负荷的时间轴（填补可能因DST产生的1小时缺失）
    temp = temp_series.reindex(df_index).interpolate(method="time").ffill().bfill()

    feat = pd.DataFrame(index=df_index)
    feat["temp"] = temp

    # 2. HDD / CDD
    feat["HDD"] = np.maximum(0, base_temp - feat["temp"])
    feat["CDD"] = np.maximum(0, feat["temp"] - base_temp)

    # 3. 滞后与滚动特征（核心防泄露：必须 shift！）
    # 预测明天 14:00，可以用到今天 14:00 的气温
    feat["temp_lag_24"] = feat["temp"].shift(24)
    # 用过去 24 小时的平均气温（shift 1 确保不含当前点）
    feat["temp_rolling_24"] = feat["temp"].rolling(24).mean().shift(1)

    # 4. 温度突变特征（今天比昨天热多少/冷多少）
    feat["temp_diff_24"] = feat["temp"] - feat["temp_lag_24"]

    return feat
