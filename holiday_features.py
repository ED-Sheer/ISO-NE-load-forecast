# holiday_features.py
import holidays
import pandas as pd


def build_holiday_features(df_index, country="US"):
    """
    输入:
        df_index : 负荷数据的 DatetimeIndex
    返回:
        DataFrame, index 与 df_index 一致
    """
    years = sorted(set(df_index.year))
    # 获取美国联邦节假日
    us_holidays = holidays.country_holidays(country, years=years)

    # 1. 判断当天是否为节假日 (转为日期比较)
    dates = df_index.date
    is_holiday = [1 if d in us_holidays else 0 for d in dates]

    feat = pd.DataFrame(index=df_index)
    feat["is_holiday"] = is_holiday

    # 2. 节前 / 节后 (按天平移)
    # 建立日级 Series
    daily_holiday = pd.Series(is_holiday, index=df_index).resample("D").max()

    # 前一天是节假日 (即节后)
    daily_next = daily_holiday.shift(-1).fillna(0)
    # 后一天是节假日 (即节前)
    daily_prev = daily_holiday.shift(1).fillna(0)

    # 映射回小时级索引
    feat["is_day_before_holiday"] = daily_prev.reindex(pd.to_datetime(dates)).values
    feat["is_day_after_holiday"] = daily_next.reindex(pd.to_datetime(dates)).values

    return feat
