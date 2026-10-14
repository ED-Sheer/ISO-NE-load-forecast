# D6_early_stopping.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings

from D1 import load_data, rolling_splits
from build_features import build_features, CATEGORICAL_FEATURES
from metrics import all_metrics
from D5_lightgbm_backtest import to_categorical, make_model

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def backtest_es(df, target='RT_Demand',
                train_hours=24*180, horizon=24, step=24,
                val_ratio=0.1, early_stopping_rounds=50):
    """
    带 early stopping 的滚动回测。
    - 从每段训练集尾部切 val_ratio 比例作为验证集
    - 验证集评估用 MAE，和训练指标一致
    - 早停保留最佳 iteration
    """
    results = []
    best_iters = []

    for i, (train_idx, test_idx) in enumerate(
            rolling_splits(df, train_hours, horizon, step)):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        feat_train = build_features(train_df, feature_set='time_lag_rolling')
        feat_test = build_features(pd.concat([train_df, test_df]),
                                    feature_set='time_lag_rolling')
        feat_test = feat_test.loc[test_idx]

        X_train_all = feat_train.dropna()
        y_train_all = train_df.loc[X_train_all.index, target]

        X_test = feat_test.dropna()
        y_test = test_df.loc[X_test.index, target]

        if len(X_test) == 0 or len(X_train_all) < 100:
            continue

        # 尾部切验证集
        n_val = max(int(len(X_train_all) * val_ratio), 48)
        X_tr = X_train_all.iloc[:-n_val]
        y_tr = y_train_all.iloc[:-n_val]
        X_val = X_train_all.iloc[-n_val:]
        y_val = y_train_all.iloc[-n_val:]

        X_tr, X_val, X_test = to_categorical_multi([X_tr, X_val, X_test],
                                                    CATEGORICAL_FEATURES)

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            n_estimators=1000,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        pred = model.predict(X_test, num_iteration=model.best_iteration_)

        m = all_metrics(
            y_true=y_test.values,
            y_pred=pred,
            y_train=y_tr.values,
            season=24,
        )
        m['split_id'] = i
        m['test_start'] = test_idx[0]
        m['best_iter'] = model.best_iteration_
        results.append(m)
        best_iters.append(model.best_iteration_)
    if len(best_iters) == 0:
        print(f">> train_hours={train_hours} 无有效切分，跳过")
        return pd.DataFrame()
    df_res = pd.DataFrame(results)
    print(f">> best_iter: mean={np.mean(best_iters):.1f}, "
          f"median={np.median(best_iters):.0f}, "
          f"min={np.min(best_iters)}, max={np.max(best_iters)}")
    return df_res


def to_categorical_multi(Xs, cat_cols):
    """对多个 DataFrame 同时做 category 编码，保证 categories 一致。"""
    Xs = [X.copy() for X in Xs]
    for col in cat_cols:
        cats = sorted(set().union(*[set(X[col].unique()) for X in Xs]))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        for X in Xs:
            if col in X.columns:
                X[col] = X[col].astype(dtype)
    return Xs


if __name__ == '__main__':
    df = load_data('.data/2024_smd_hourly.xlsx')

    print(">>> 跑 early stopping 版 ...")
    res_es = backtest_es(df)

    print("\n=== early stopping 结果 (滚动回测 mean) ===")
    for k in ['MAE', 'RMSE', 'MAPE', 'MASE']:
        print(f"{k:>5}: mean={res_es[k].mean():.4f}, median={res_es[k].median():.4f}")

    # 和 D5 无早停对比
    res_full = pd.read_csv('results_d5_full.csv')
    print("\n=== 对比 D5 (无早停) ===")
    print(f"{'指标':>5} | {'D5 无早停':>12} | {'D6 早停':>12} | {'变化':>10}")
    for k in ['MAE', 'RMSE', 'MAPE', 'MASE']:
        v5 = res_full[k].mean()
        v6 = res_es[k].mean()
        print(f"{k:>5} | {v5:>12.4f} | {v6:>12.4f} | {v6-v5:>+10.4f}")

    # 按星期几分组对比
    for r in (res_es,):
        r['dow'] = pd.to_datetime(r['test_start']).dt.dayofweek
    res_full['test_start'] = pd.to_datetime(res_full['test_start'])
    res_full['dow'] = res_full['test_start'].dt.dayofweek

    dow_cmp = pd.DataFrame({
        'd5_no_es': res_full.groupby('dow')['MAPE'].mean(),
        'd6_es':    res_es.groupby('dow')['MAPE'].mean(),
    }, index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    print("\n=== 按星期几分组 ===")
    print(dow_cmp.round(3))

    # 逐段折线对比
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(res_full['test_start'], res_full['MAPE'],
            marker='o', markersize=2, linewidth=0.6,
            color='#60A5FA', alpha=0.7, label='D5 no early stopping')
    ax.plot(pd.to_datetime(res_es['test_start']), res_es['MAPE'],
            marker='o', markersize=2, linewidth=0.6,
            color='#2563EB', label='D6 early stopping')
    ax.axhline(5.9522, color='#DC2626', linestyle='--',
               linewidth=1.2, label='naive = 5.95%')
    ax.set_title('D6 Early Stopping: Per-window MAPE', fontsize=13)
    ax.set_xlabel('Test start date')
    ax.set_ylabel('MAPE (%)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d6_es_per_window.png', dpi=150, bbox_inches='tight')
    plt.show()

    # best_iter 分布
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(res_es['best_iter'], bins=30, color='#2563EB', alpha=0.8)
    ax.axvline(res_es['best_iter'].median(), color='#DC2626',
               linestyle='--', label=f"median = {res_es['best_iter'].median():.0f}")
    ax.set_xlabel('best_iteration')
    ax.set_ylabel('Count')
    ax.set_title('D6 best_iteration distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d6_best_iter.png', dpi=150, bbox_inches='tight')
    plt.show()

    res_es.to_csv('results_d6_es.csv', index=False)
    print("\n>> 已保存 results_d6_es.csv 和 plots/d6_*.png")
