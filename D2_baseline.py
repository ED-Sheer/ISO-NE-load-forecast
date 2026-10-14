import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from D1 import load_data, train_test_split_ts   # 复用 D1 的 load 和切分
from metrics import all_metrics, mae, rmse, mape, mase

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------
# 自检：用人工算得出的数据验证四个指标
# ------------------------------------------------------------
def self_check():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 180.0, 330.0, 360.0])

    # 手工核对：
    # |err| = [10, 20, 30, 40] → MAE = 25
    # err^2 = [100, 400, 900, 1600] → mean = 750 → RMSE ≈ 27.386
    # |err|/y_true = [0.1, 0.1, 0.1, 0.1] → MAPE = 10%
    assert abs(mae(y_true, y_pred) - 25.0) < 1e-9
    assert abs(rmse(y_true, y_pred) - np.sqrt(750)) < 1e-9
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-9
    print(">> 自检通过: MAE=25, RMSE≈27.386, MAPE=10%")

    # 除零保护：y_true 含 0 时不应崩、不应把 0 当分母
    y_true2 = np.array([0.0, 100.0])
    y_pred2 = np.array([50.0, 90.0])
    v = mape(y_true2, y_pred2)
    assert abs(v - 10.0) < 1e-9, f"MAPE 除零保护失败，得到 {v}"
    print(">> 除零保护通过: 只对 y_true!=0 的点计算 MAPE")


# ------------------------------------------------------------
# 朴素预测：pred[t] = y[t - lag]
# 关键：先拼 train+test 再 shift，测试集开头也能借到训练集尾巴
# ------------------------------------------------------------
def naive_forecast(y_all, test_index, lag=24):
    """
    在完整序列上算 shift(lag)，只返回 test_index 那一段。
    y_all     : Series，训练+测试拼接后的完整目标序列（按时间排序）
    test_index: DatetimeIndex，测试集时间戳
    lag       : 24 表示“昨天同一时刻”
    """
    shifted = y_all.shift(lag)
    pred = shifted.loc[test_index]
    return pred


# ------------------------------------------------------------
# 画图：真实值 vs 朴素预测
# ------------------------------------------------------------
def plot_naive(y_true, y_pred, title="Naive Forecast vs Actual"):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(y_true.index, y_true.values, color="#2563EB",
            linewidth=0.9, label="Actual")
    ax.plot(y_pred.index, y_pred.values, color="#DC2626",
            linewidth=0.9, linestyle="--", label="Naive (lag=24)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("RT_Demand (MW)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/naive_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(">> 已保存: plots/naive_forecast.png")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1) 自检
    self_check()

    # 2) 加载数据（复用 D1）
    df = load_data(".data/2024_smd_hourly.xlsx")
    y = df["RT_Demand"].dropna()

    # 3) 时序切分（沿用 D1 的切分点）
    split_date = pd.Timestamp("2024-07-01")
    X = pd.DataFrame(index=y.index)
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y, split_date)

    # 4) 朴素预测：先拼 train+test，再 shift(24)，最后只取 test 段
    y_all = pd.concat([y_train, y_test]).sort_index()
    y_pred = naive_forecast(y_all, y_test.index, lag=24)

    # 对齐：丢掉 y_pred 中的 NaN（理论上不会有，但保险）
    valid = y_pred.notna()
    y_test_aligned = y_test.loc[valid]
    y_pred_aligned = y_pred.loc[valid]

    print(f"\n测试段长度: {len(y_test_aligned)}")
    print(f"朴素预测 NaN 数: {(~valid).sum()}")

    # 5) 计算四个指标
    result = all_metrics(
        y_true=y_test_aligned.values,
        y_pred=y_pred_aligned.values,
        y_train=y_train.values,
        season=24,
    )
    print("\n=== 朴素预测 (lag=24) 基线 ===")
    for k, v in result.items():
        print(f"{k:>5}: {v:.4f}")

    # 6) 画图
    plot_naive(y_test_aligned, y_pred_aligned)

    # 7) 记录到 notes.md 供 D3/D5/D6 引用
    with open("notes.md", "a", encoding="utf-8") as f:
        f.write(
            f"\n## D2 朴素预测基线 (lag=24)\n"
            f"- 切分点: {split_date.date()}\n"
            f"- 训练集: {len(y_train)} 条\n"
            f"- 测试集: {len(y_test_aligned)} 条\n"
            f"- MAE:  {result['MAE']:.2f} MW\n"
            f"- RMSE: {result['RMSE']:.2f} MW\n"
            f"- MAPE: {result['MAPE']:.4f} %\n"
            f"- MASE: {result['MASE']:.4f}\n"
        )
    print("\n>> 已追加到 notes.md")