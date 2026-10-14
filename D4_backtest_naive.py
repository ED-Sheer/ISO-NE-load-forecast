import pandas as pd
import matplotlib.pyplot as plt

from D1 import load_data , rolling_splits
from metrics import all_metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------
# backtest_naive() —— 用 naive 做滚动回测，验证切分
# ------------------------------------------------------------
def backtest_naive(df, target='RT_Demand',
                   train_hours=24*180, horizon=24, step=24, lag=24):
    """
    专门用于 D4 验证：每次用 naive（pred[t] = y[t-24]）预测，
    不依赖 build_features，不依赖任何模型。
    """
    results = []
    for i, (train_idx, test_idx) in enumerate(
            rolling_splits(df, train_hours, horizon, step)):

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        # 拼接后 shift，取 test 段 —— 和 D2/D1 一样的套路
        y_all = pd.concat([train_df[target], test_df[target]]).sort_index()
        pred = y_all.shift(lag).loc[test_idx]

        y_true = test_df[target].loc[pred.dropna().index]
        pred = pred.dropna()
        if len(y_true) == 0:
            continue

        m = all_metrics(
            y_true=y_true.values,
            y_pred=pred.values,
            y_train=train_df[target].values,
            season=24,
        )
        m['split_id'] = i
        m['test_start'] = test_idx[0]
        m['test_end'] = test_idx[-1]
        m['n_test'] = len(y_true)
        results.append(m)

    return pd.DataFrame(results)


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(".data/2024_smd_hourly.xlsx")

    # ---------- 1) 切分边界自检 ----------
    splits = rolling_splits(df, train_hours=24*180, horizon=24, step=24)
    print("\n=== 滚动切分概览 ===")
    print(f"总切分数: {len(splits)}")
    print(f"第一次 train: {splits[0][0][0]} ~ {splits[0][0][-1]}")
    print(f"第一次 test : {splits[0][1][0]} ~ {splits[0][1][-1]}")
    print(f"最后一次 train: {splits[-1][0][0]} ~ {splits[-1][0][-1]}")
    print(f"最后一次 test : {splits[-1][1][0]} ~ {splits[-1][1][-1]}")

    # 关键断言 1：每次切分 train < test
    for i, (tr, te) in enumerate(splits):
        assert tr.max() < te.min(), f"第 {i} 次切分 train/test 重叠"
    print(">> 断言通过: 所有切分 train 严格早于 test，无未来泄露")

    # 关键断言 2：训练窗口长度固定（滑动窗口的特征）
    train_lens = set(len(tr) for tr, _ in splits)
    test_lens = set(len(te) for _, te in splits)
    print(f">> 训练窗口长度集合: {train_lens}  (应为 {{4320}})")
    print(f">> 测试窗口长度集合: {test_lens}   (应为 {{24}})")
    assert train_lens == {24*180}, "训练窗口长度不固定，不是滑动窗口"
    assert test_lens == {24}, "测试窗口长度不等于 horizon"

    # 关键断言 3：相邻切分不重叠、不跳空
    for i in range(1, len(splits)):
        prev_test_end = splits[i-1][1][-1]
        this_test_start = splits[i][1][0]
        delta = (this_test_start - prev_test_end).total_seconds() / 3600
        assert delta == 1, f"第 {i} 次和上一次测试段间隔 {delta} 小时，应恰好 1"
    print(">> 断言通过: 相邻测试段首尾相接，无跳空")

    # ---------- 2) 滚动回测 naive ----------
    results_df = backtest_naive(df, target='RT_Demand',
                                 train_hours=24*180, horizon=24,
                                 step=24, lag=24)

    print(f"\n=== 滚动回测 (naive, lag=24) ===")
    print(f"总段数: {len(results_df)}")
    print(f"MAPE 分布: mean={results_df['MAPE'].mean():.4f}%, "
          f"median={results_df['MAPE'].median():.4f}%, "
          f"min={results_df['MAPE'].min():.4f}%, "
          f"max={results_df['MAPE'].max():.4f}%")

    # ---------- 3) 和 D2 对比 ----------
    print(f"\n=== 和 D2 单次切分对比 ===")
    d2 = {"MAE": 802.22, "RMSE": 1130.25, "MAPE": 5.8888, "MASE": 0.9702}
    print(f"{'指标':>5} | {'D2 单次':>12} | {'D4 滚动(均)':>14} | {'差异':>10}")
    for k in ["MAE", "RMSE", "MAPE", "MASE"]:
        v = results_df[k].mean()
        print(f"{k:>5} | {d2[k]:>12.4f} | {v:>14.4f} | {v - d2[k]:>+10.4f}")

    # ---------- 4) 逐段 MAPE 图 ----------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(results_df['test_start'], results_df['MAPE'],
            marker='o', markersize=2.5, color="#2563EB", linewidth=0.7)
    ax.axhline(d2['MAPE'], color="#DC2626", linestyle="--",
               label=f"D2 单次切分 = {d2['MAPE']}%")
    ax.set_title("D4 Rolling Backtest (naive): Per-window MAPE", fontsize=12)
    ax.set_xlabel("Test start date")
    ax.set_ylabel("MAPE (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/d4_naive_per_window_mape.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ---------- 5) 保存 ----------
    results_df.to_csv("results_d4_naive.csv", index=False)
    print("\n>> 已保存: results_d4_naive.csv 和 plots/d4_naive_per_window_mape.png")
    # df_r = pd.read_csv("results_d4_naive.csv")
    # df_r["test_start"] = pd.to_datetime(df_r["test_start"])
    # df_r["dow"] = df_r["test_start"].dt.dayofweek  # 0=周一, 5=周六, 6=周日
    #
    # # 按星期几分组看 MAPE
    # print(df_r.groupby("dow")["MAPE"].agg(["mean", "median", "max", "count"]))
    #
    # # 看 MAPE 最高的 10 段
    # print(df_r.sort_values("MAPE", ascending=False).head(10)[["test_start", "MAPE"]])
