# D6_monthly_curve.py
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    res_full = pd.read_csv('results_d5_full.csv')
    res_time = pd.read_csv('results_d5_time_only.csv')

    for r in (res_full, res_time):
        r['test_start'] = pd.to_datetime(r['test_start'])
        r['month'] = r['test_start'].dt.month

    # 逐月 MAPE
    by_month_full = res_full.groupby('month')['MAPE'].agg(['mean', 'median', 'std'])
    by_month_time = res_time.groupby('month')['MAPE'].agg(['mean', 'median', 'std'])
    print("=== full 逐月 ===")
    print(by_month_full.round(3))
    print("\n=== time_only 逐月 ===")
    print(by_month_time.round(3))

    # 画折线
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(by_month_time.index, by_month_time['mean'],
            marker='o', markersize=8, linewidth=2,
            color='#60A5FA', label='LightGBM (time only)')
    ax.plot(by_month_full.index, by_month_full['mean'],
            marker='s', markersize=8, linewidth=2,
            color='#2563EB', label='LightGBM (time+lag+rolling)')
    ax.axhline(5.9522, color='#DC2626', linestyle='--',
               linewidth=1.3, label='naive baseline = 5.95%')

    # 误差带 (mean ± std)
    ax.fill_between(by_month_full.index,
                    by_month_full['mean'] - by_month_full['std'],
                    by_month_full['mean'] + by_month_full['std'],
                    color='#2563EB', alpha=0.12)

    ax.set_xticks(range(6, 13))
    ax.set_xticklabels([f'{m}月' for m in range(6, 13)])
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_title('D6 Monthly MAPE: time only vs full', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/d6_monthly_mape.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n>> 已保存: plots/d6_monthly_mape.png")
