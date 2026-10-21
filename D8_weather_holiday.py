# D8_weather_holiday.py
import pandas as pd
from D1 import load_data
from D5_lightgbm_backtest import backtest
from fetch_weather import fetch_weather
from weather_features import build_weather_features
from holiday_features import build_holiday_features

if __name__ == '__main__':
    print("加载数据...")
    df = load_data(".data/2024_smd_hourly.xlsx")

    print("拉取天气...")
    temp = fetch_weather(start="2024-01-01", end="2024-12-31")
    weather_df = build_weather_features(df.index, temp)
    holiday_df = build_holiday_features(df.index)

    print("\n>>> 1. 跑 D5 基线 (仅 lag/rolling) ...")
    res_base = backtest(df, feature_set='time_lag_rolling')
    print(f"MAPE: {res_base['MAPE'].mean():.4f}%")

    print("\n>>> 2. 跑 加天气 ...")
    res_w = backtest(df, feature_set='time_lag_rolling_weather_holiday',
                     weather_df=weather_df, holiday_df=None)
    print(f"MAPE: {res_w['MAPE'].mean():.4f}%")

    print("\n>>> 3. 跑 加天气 + 节假日 ...")
    res_wh = backtest(df, feature_set='time_lag_rolling_weather_holiday',
                      weather_df=weather_df, holiday_df=holiday_df)
    print(f"MAPE: {res_wh['MAPE'].mean():.4f}%")

    # 提取 7-8 月进行对比
    for name, r in [('Baseline', res_base), ('Weather', res_w), ('Weather+Holiday', res_wh)]:
        r['month'] = pd.to_datetime(r['test_start']).dt.month
        summer_mape = r[r['month'].isin([7, 8])]['MAPE'].mean()
        print(f"{name:>20} 夏季(7-8月) MAPE: {summer_mape:.4f}%")

    print(res_wh.mean())