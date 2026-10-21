# D5_lightgbm_backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from D1 import load_data, rolling_splits
from build_features import build_features, CATEGORICAL_FEATURES
from metrics import all_metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def to_categorical(X_train, X_test, cat_cols):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in cat_cols:
        if col not in X_train.columns:
            continue
        cats = sorted(set(X_train[col].unique()) | set(X_test[col].unique()))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        X_train[col] = X_train[col].astype(dtype)
        X_test[col] = X_test[col].astype(dtype)
    return X_train, X_test


def make_model():
    return lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_estimators=500,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )


def backtest(df, feature_set, target='RT_Demand',
             train_hours=24*180, horizon=24, step=24, weather_df=None, holiday_df=None):
    """
    滚动回测。每次：
      - train_df: 滑动的 180 天窗口
      - test_df : 紧接着的 24 小时
      - 特征: train 用自己的窗口算；test 用 train+test 拼接算（lag 能借训练集尾巴）
    """
    results = []
    for i, (train_idx, test_idx) in enumerate(
            rolling_splits(df, train_hours, horizon, step)):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        feat_train = build_features(train_df, feature_set=feature_set,
                                    weather_df=weather_df, holiday_df=holiday_df)
        feat_test = build_features(pd.concat([train_df, test_df]), feature_set=feature_set,
                                   weather_df=weather_df, holiday_df=holiday_df)
        feat_test = feat_test.loc[test_idx]

        # 对齐 dropna
        X_train = feat_train.dropna()
        y_train = train_df.loc[X_train.index, target]

        X_test = feat_test.dropna()
        y_test = test_df.loc[X_test.index, target]

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        X_train, X_test = to_categorical(X_train, X_test, CATEGORICAL_FEATURES)

        model = make_model()
        model.fit(X_train, y_train,
                  categorical_feature=CATEGORICAL_FEATURES)
        pred = model.predict(X_test)

        m = all_metrics(
            y_true=y_test.values,
            y_pred=pred,
            y_train=y_train.values,
            season=24,
        )
        m['split_id'] = i
        m['test_start'] = test_idx[0]
        m['n_test'] = len(X_test)
        results.append(m)

    return pd.DataFrame(results)


if __name__ == '__main__':
    df = load_data('.data/2024_smd_hourly.xlsx')

    # ---------- 1) time_only ----------
    print(">>> 跑 time_only ...")
    res_time = backtest(df, feature_set='time_only')
    print(f"段数: {len(res_time)}")
    print("=== time_only (hour/dayofweek/is_weekend) ===")
    for k in ['MAE', 'RMSE', 'MAPE', 'MASE']:
        print(f"{k:>5}: mean={res_time[k].mean():.4f}, median={res_time[k].median():.4f}")

    # ---------- 2) time + lag + rolling ----------
    print("\n>>> 跑 time_lag_rolling ...")
    res_full = backtest(df, feature_set='time_lag_rolling')
    print(f"段数: {len(res_full)}")
    print("=== time + lag + rolling ===")
    for k in ['MAE', 'RMSE', 'MAPE', 'MASE']:
        print(f"{k:>5}: mean={res_full[k].mean():.4f}, median={res_full[k].median():.4f}")

    # ---------- 3) 三方对比 ----------
    d4_naive = {'MAE': 812.65, 'RMSE': 958.23, 'MAPE': 5.9522, 'MASE': 0.9056}
    print("\n=== 三方对比 (滚动回测口径, mean) ===")
    header = f"{'指标':>5} | {'naive':>10} | {'time_only':>12} | {'full':>10} | {'full-naive':>11}"
    print(header)
    print("-" * len(header))
    for k in ['MAE', 'RMSE', 'MAPE', 'MASE']:
        v1 = res_time[k].mean()
        v2 = res_full[k].mean()
        print(f"{k:>5} | {d4_naive[k]:>10.4f} | {v1:>12.4f} | {v2:>10.4f} | {v2 - d4_naive[k]:>+11.4f}")

    # ---------- 4) 柱状图：MAPE 三方对比 ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['naive\n(lag=24)', 'LightGBM\ntime only', 'LightGBM\ntime+lag+rolling']
    mapes = [d4_naive['MAPE'], res_time['MAPE'].mean(), res_full['MAPE'].mean()]
    colors = ['#94A3B8', '#60A5FA', '#2563EB']
    bars = ax.bar(labels, mapes, color=colors, width=0.55)
    for bar, v in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08,
                f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_title('D5 Rolling Backtest: MAPE Comparison', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/d5_mape_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---------- 5) 逐段 MAPE 折线对比 ----------
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(res_time['test_start'], res_time['MAPE'],
            marker='o', markersize=2, linewidth=0.6,
            color='#60A5FA', alpha=0.7, label='time only')
    ax.plot(res_full['test_start'], res_full['MAPE'],
            marker='o', markersize=2, linewidth=0.6,
            color='#2563EB', label='time + lag + rolling')
    ax.axhline(d4_naive['MAPE'], color='#DC2626', linestyle='--',
               linewidth=1.2, label=f"naive = {d4_naive['MAPE']}%")
    ax.set_title('D5 Rolling Backtest: Per-window MAPE', fontsize=13)
    ax.set_xlabel('Test start date')
    ax.set_ylabel('MAPE (%)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d5_per_window_mape.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---------- 6) 按星期几分组对比 ----------
    for r in (res_time, res_full):
        r['dow'] = pd.to_datetime(r['test_start']).dt.dayofweek
    dow_cmp = pd.DataFrame({
        'naive_est': [7.97, 6.27, 4.37, 4.52, 5.82, 6.42, 6.19],  # 从 D4 表复制
        'time_only': res_time.groupby('dow')['MAPE'].mean().values,
        'full':      res_full.groupby('dow')['MAPE'].mean().values,
    }, index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    print("\n=== 按星期几分组 MAPE ===")
    print(dow_cmp.round(3))

    dow_cmp.plot(kind='bar', figsize=(10, 4.5),
                 color=['#94A3B8', '#60A5FA', '#2563EB'])
    plt.ylabel('MAPE (%)')
    plt.title('D5 MAPE by Day of Week', fontsize=13)
    plt.xticks(rotation=0)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/d5_by_dow.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---------- 7) 按月份分组 ----------
    for r in (res_time, res_full):
        r['month'] = pd.to_datetime(r['test_start']).dt.month
    month_cmp = pd.DataFrame({
        'time_only': res_time.groupby('month')['MAPE'].mean(),
        'full':      res_full.groupby('month')['MAPE'].mean(),
    })
    print("\n=== 按月份分组 MAPE ===")
    print(month_cmp.round(3))

    month_cmp.plot(figsize=(10, 4.5), marker='o',
                   color=['#60A5FA', '#2563EB'])
    plt.ylabel('MAPE (%)')
    plt.title('D5 MAPE by Month', fontsize=13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d5_by_month.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ---------- 8) 保存 ----------
    res_time.to_csv('results_d5_time_only.csv', index=False)
    res_full.to_csv('results_d5_full.csv', index=False)
    print("\n>> 已保存 results_d5_*.csv 和 plots/d5_*.png")