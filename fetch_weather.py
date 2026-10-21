# fetch_weather.py（替换成这个版本）
import requests
import pandas as pd

def fetch_weather(lat=42.3584, lon=-71.0598,
                  start="2024-01-01", end="2024-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    # data["time"] 是形如 "2024-01-01T00:00" 的本地字符串，直接转 pandas 即可
    temp = pd.Series(
        data["temperature_2m"],
        index=pd.to_datetime(data["time"]),
        name="temp",
    )

    # DST 去重（和 load_data 保持一致）
    temp = temp[~temp.index.duplicated(keep="first")]
    temp = temp.sort_index()
    temp = temp.interpolate(method="time").ffill().bfill()
    return temp