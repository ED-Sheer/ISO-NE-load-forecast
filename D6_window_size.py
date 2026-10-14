# D6_window_size.py
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from D1 import load_data
from D6_early_stopping import backtest_es

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    df = load_data('.data/2024_smd_hourly.xlsx')

    windows_days = [90, 180, 365]
    all_res = {}

    for wd in windows_days:
        print(f"\n>>> train_hours = {wd*24}")
        res = backtest_es(df, train_hours=wd*24)
        if len(res) == 0:
            print(f"   window={wd}d 无有效切分，跳过")
            continue
        all_res[wd] = res
        print(f"   MAPE mean = {res['MAPE'].mean():.4f}, "
              f"median = {res['MAPE'].median():.4f}")

    # 对比表
    print("\n=== 不同训练窗口对比 ===")
    print(f"{'窗口天数':>8} | {'MAPE mean':>10} | {'MAPE median':>12} | {'RMSE mean':>10}")
    for wd, r in all_res.items():
        print(f"{wd:>8} | {r['MAPE'].mean():>10.4f} | "
              f"{r['MAPE'].median():>12.4f} | {r['RMSE'].mean():>10.4f}")

    # 逐段折线
    fig, ax = plt.subplots(figsize=(14, 4.5))
    colors = {90: '#94A3B8', 180: '#60A5FA', 365: '#2563EB'}
    for wd, r in all_res.items():
        r['test_start'] = pd.to_datetime(r['test_start'])
        ax.plot(r['test_start'], r['MAPE'],
                marker='o', markersize=2, linewidth=0.6,
                color=colors[wd], label=f'window={wd}d')
    ax.axhline(5.9522, color='#DC2626', linestyle='--',
               linewidth=1.2, label='naive = 5.95%')
    ax.set_title('D6 Train Window Size Comparison', fontsize=13)
    ax.set_xlabel('Test start date')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d6_window_size.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 保存
    for wd, r in all_res.items():
        r.to_csv(f'results_d6_window{wd}.csv', index=False)
    print("\n>> 已保存 results_d6_window*.csv 和 plots/d6_window_size.png")
