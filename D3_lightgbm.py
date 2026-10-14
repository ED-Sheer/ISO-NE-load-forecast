# D3_lightgbm.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from D1 import load_data, train_test_split_ts
from build_features import build_features, CATEGORICAL_FEATURES
from metrics import all_metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------
# 类别特征转换
# ------------------------------------------------------------
def to_categorical(X_train, X_test, cat_cols):
    """
    把 cat_cols 里的列转成 pandas category dtype，
    并保证 train/test 的 categories 完全一致（LightGBM 要求）。
    """
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


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1) 加载数据
    df = load_data(".data/2024_smd_hourly.xlsx")
    y = df["RT_Demand"].dropna()

    # 2) 构造特征（只加时间特征）
    X_all = build_features(df.loc[y.index])

    # 3) 时序切分，沿用 D1/D2 的切分点
    split_date = pd.Timestamp("2024-07-01")
    X_train, X_test, y_train, y_test = train_test_split_ts(X_all, y, split_date)

    # 4) 类别特征处理
    X_train, X_test = to_categorical(X_train, X_test, CATEGORICAL_FEATURES)

    print("\n=== 特征矩阵 ===")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"列: {list(X_train.columns)}")
    print(f"dtypes:\n{X_train.dtypes}")

    # 5) 训练 LightGBM
    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_estimators=500,
        verbose=-1,
        random_state=42,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    # 6) 预测 + 评估
    pred = model.predict(X_test)

    result = all_metrics(
        y_true=y_test.values,
        y_pred=pred,
        y_train=y_train.values,
        season=24,
    )

    print("\n=== D3 LightGBM (只加时间特征) ===")
    for k, v in result.items():
        print(f"{k:>5}: {v:.4f}")

    # 7) 和 D2 基线对比
    baseline = {"MAE": 802.22, "RMSE": 1130.25, "MAPE": 5.8888, "MASE": 0.9702}
    print("\n=== 与 D2 朴素预测基线对比 ===")
    print(f"{'指标':>5} | {'naive':>10} | {'LightGBM':>10} | {'变化':>10}")
    for k in ["MAE", "RMSE", "MAPE", "MASE"]:
        delta = result[k] - baseline[k]
        print(f"{k:>5} | {baseline[k]:>10.4f} | {result[k]:>10.4f} | {delta:>+10.4f}")

    # 8) 特征重要性
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    imp.plot(kind="barh", ax=ax, color="#2563EB")
    ax.set_title("LightGBM Feature Importance (D3: time only)", fontsize=12)
    ax.set_xlabel("Importance (split count)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("plots/d3_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 9) 预测 vs 实际（只看测试段前 7 天）
    n_show = 24 * 7
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.index[:n_show], y_test.values[:n_show],
            color="#2563EB", linewidth=1.0, label="Actual")
    ax.plot(y_test.index[:n_show], pred[:n_show],
            color="#DC2626", linewidth=1.0, linestyle="--", label="LightGBM")
    ax.set_title("D3 LightGBM: Test first 7 days", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("RT_Demand (MW)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/d3_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 10) 记录到独立文件，避免 notes.md 无限追加
    with open("notes_d3.md", "w", encoding="utf-8") as f:
        f.write(
            f"# D3 LightGBM (只加时间特征)\n\n"
            f"- 切分点: {split_date.date()}\n"
            f"- 特征: {list(X_train.columns)}\n"
            f"- 参数: {params}\n\n"
            f"| 指标 | naive 基线 | LightGBM | 变化 |\n"
            f"|---|---|---|---|\n"
            f"| MAE  | {baseline['MAE']:.4f}  | {result['MAE']:.4f}  | {result['MAE']-baseline['MAE']:+.4f} |\n"
            f"| RMSE | {baseline['RMSE']:.4f} | {result['RMSE']:.4f} | {result['RMSE']-baseline['RMSE']:+.4f} |\n"
            f"| MAPE | {baseline['MAPE']:.4f}% | {result['MAPE']:.4f}% | {result['MAPE']-baseline['MAPE']:+.4f} |\n"
            f"| MASE | {baseline['MASE']:.4f} | {result['MASE']:.4f} | {result['MASE']-baseline['MASE']:+.4f} |\n"
        )
    print("\n>> 已写 notes_d3.md")