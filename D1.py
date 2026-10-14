import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.seasonal import STL

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
os.makedirs('plots', exist_ok=True)

# ----------------------------
# rolling_splits() - 滚动窗口切分
# ----------------------------
def rolling_splits(df, train_hours=24*180, horizon=24, step=24):
    """
    生成滚动窗口切分，返回 (train_idx, test_idx) 列表
    """
    splits = []
    n = len(df)
    if train_hours + horizon > n:
        raise ValueError(
            f"train_hours({train_hours}) + horizon({horizon}) > "
            f"数据长度({n})，无法生成任何切分。"
            f"请减小 train_hours，或增加数据。"
        )
    start = train_hours
    while start + horizon <= n:
        train_idx = df.index[start - train_hours : start]
        test_idx = df.index[start:start + horizon]
        splits.append((train_idx, test_idx))
        start += step
    return splits

def backtest(df, build_features, model_factory,
             train_hours=24*180, horizon=24, step=24, target="RT_Demand"):
    results = []
    for train_idx, test_idx in rolling_splits(df, train_hours, horizon, step):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        feat_train = build_features(train_df)
        feat_test = build_features(pd.concat([train_df, test_df]))  # 关键：lag/rolling要用连续序列算
        feat_test = feat_test.loc[test_idx].dropna()

        if len(feat_test) == 0:
            continue

        X_train = feat_train.drop(columns=[c for c in [target] if c in feat_train]).dropna()
        y_train = train_df.loc[X_train.index, target]
        X_test = feat_test.drop(columns=[c for c in [target] if c in feat_test])
        y_test = test_df.loc[X_test.index, target]

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "test_start": test_idx[0],
            "MAE": float(np.mean(np.abs(y_test - pred))),
            "RMSE": float(np.sqrt(np.mean((y_test - pred) ** 2))),
            "MAPE": float(np.mean(np.abs((y_test - pred) / y_test.replace(0, np.nan))) * 100),
        })
    return pd.DataFrame(results)

# ------------------------------
# load_data() - 加载负荷数据
# ------------------------------
def load_data(path):
    """
    读取数据，处理夏令时异常值，返回以 Datatime 为索引的 DataFrame。
    :param path: Excel 文件路径
    :return: DataFrame, index 为 Datetime, 列为 ['RT_Demand']等
    """
    # 1. 读取 ISO NE CA 的 Sheet
    df = pd.read_excel(path,sheet_name = 'ISO NE CA')

    # 2. 数据清理
    df['Hr_End'] = df['Hr_End'].astype(str).str.strip()
    df.loc[df['Hr_End'] == '02X', 'Hr_End'] = '02'

    # 3. 组合 Date 和 Hr_End 成完整时间戳
    df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hr_End'].astype(int) - 1, unit = 'h')
    df = df.set_index('Datetime')

    # 4. 清理多余列
    df = df.drop(columns = ['Date', 'Hr_End'])

    # 确保 RT_Demand 是数值类型
    df['RT_Demand'] = pd.to_numeric(df['RT_Demand'], errors='coerce')

    # 去重：02X→02 可能产生重复时间戳，保留第一次出现的
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    gaps = df.index.to_series().diff().value_counts()
    print("时间间隔分布（前5）:")
    print(gaps.head())

    return df

# --------------------------------------------
# train_test_split_ts() - 时序切分（禁止 shuffle）
# --------------------------------------------
def train_test_split_ts(X, y, split_date):
    """
    按时间点切分训练集/测试集。测试集必须在训练集之后，不允许随机打乱。
    :param X: DataFrame, 特征矩阵 (index 为时间戳)
    :param y: Series, 目标变量 (index 为时间戳)
    :param split_date: Timestamp, 切分时间点（包含该时刻及之前的归为训练集）
    :return: X_train, X_test, y_train, y_test
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("X.index 必须是 DatetimeIndex，请先 pd.to_datetime 并 set_index")

    # 兜底：如果 X 和 y 长度不一致，取共同索引对齐
    if len(X) != len(y):
        n_x, n_y = len(X), len(y)
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        print(f"WARNING: X({n_x}) 和 y({n_y}) 长度不一致，已取共同索引对齐 → {len(common)} 条")

    # 去重索引（防止 02X→02 等产生重复时间戳）
    dup_mask = X.index.duplicated(keep='first')
    if dup_mask.any():
        X = X[~dup_mask]
        y = y[~dup_mask]
        print(f"WARNING: 检测到 {dup_mask.sum()} 个重复时间戳，已去除")

    # 排序 + 对齐
    X = X.sort_index()
    y = y.loc[X.index]

    # 去掉 y 为 NaN 的行
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    split_date = pd.to_datetime(split_date)

    train_mask = X.index <= split_date
    test_mask = X.index > split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"切分后存在空集，检查 split_date={split_date} 是否落在数据范围内")

    assert X_train.index.max() < X_test.index.min(), "训练集包含测试之后的时间，存在未来泄露"

    print(f"训练集: {len(X_train)} 条 ({X_train.index.min()} ~ {X_train.index.max()})")
    print(f"测试集: {len(X_test)} 条 ({X_test.index.min()} ~ {X_test.index.max()})")

    return X_train, X_test, y_train, y_test

# ----------------------------------
# plot_split_diagram() - 画出切分示意图
# ----------------------------------
def plot_split_diagram(X_train, X_test, y, split_date):
    """
    画一张图：训练集绿色背景，测试集红色背景，蓝色为真实负荷曲线，
    黑色虚线标记切分点。
    :param X_train: 训练数据
    :param X_test: 测试数据
    :param y: 预测数据
    :param split_date: 切分日期
    :return: None
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # 训练集背景
    ax.axvspan(X_train.index.min(), X_train.index.max(),
               alpha=0.15, color='green', label='Training')

    # 测试集背景
    ax.axvspan(X_test.index.min(), X_test.index.max(),
               alpha=0.15, color='red', label='Test')

    # 真实负荷曲线
    y.plot(ax=ax, color='#2563EB', linewidth=0.8)

    # 切分点标记
    ax.axvline(x=split_date, color='#DC2626', linestyle='--',
               linewidth=1.5, label=f'Split: {split_date.date()}')

    ax.set_title('Time Series Train/Test Split', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('RT_Demand (MW)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/split_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(">> 切分示意图已保存: plots/split_diagram.png")

# ----------------------------
# plot_load_curve() - 画负荷曲线
# ----------------------------
def plot_load_curve(df, column = 'RT_Demand'):
    """
    画原始负荷曲线。
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    df[column].plot(ax=ax, color='#2563EB', linewidth=0.8)
    ax.set_title('ISO-NE Hourly Load', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('MW', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/load_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(">> 负荷曲线已保存: plots/load_curve.png")

def plot_stl_intraday(df, target="RT_Demand", period=24):
    s = df[target].dropna()
    res = STL(s, period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(s); axes[0].set_title("Observed")
    axes[1].plot(res.trend); axes[1].set_title("Trend")
    axes[2].plot(res.seasonal); axes[2].set_title("Seasonal (intraday 24h)")
    axes[3].plot(res.resid); axes[3].set_title("Residual")
    plt.tight_layout(); plt.savefig("plots/stl_intraday.png", dpi=120); plt.show()

def plot_stl_yearly(df, target="RT_Demand"):
    daily = df[target].resample("D").mean().dropna()
    period = 365 if len(daily) >= 730 else 181   # 只有一年数据时用181近似半年周期
    res = STL(daily, period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(daily); axes[0].set_title("Daily Mean")
    axes[1].plot(res.trend); axes[1].set_title("Trend")
    axes[2].plot(res.seasonal); axes[2].set_title(f"Seasonal (period={period})")
    axes[3].plot(res.resid); axes[3].set_title("Residual")
    plt.tight_layout(); plt.savefig("plots/stl_yearly.png", dpi=120); plt.show()

# -----------------------------------
# plot_stl_decomposition() - 季节分解图
# -----------------------------------
def plot_stl_decomposition(df, column='RT_Demand', period=24 * 7 + 1):
    """
        STL 季节分解: Observed / Trend / Seasonal / Residual 四张子图。

        参数:
            df: DataFrame
            column: 要分解的列名
            period: 季节周期长度（小时级数据通常用 24*7=168 表示周季节）
        """
    series = df[column].dropna()
    # STL 分解
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    result.observed.plot(ax=axes[0], title='Observed (原始)', color='#2563EB')
    result.trend.plot(ax=axes[1], title='Trend (趋势)', color='#DC2626')
    result.seasonal.plot(ax=axes[2], title='Seasonal (季节)', color='#059669')
    result.resid.plot(ax=axes[3], title='Residual (残差)', color='#D97706')

    for ax, label in zip(axes, ['MW', 'MW', 'MW', 'MW']):
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('MW', fontsize=10)
    axes[3].set_xlabel('Date', fontsize=11)

    plt.suptitle('STL Seasonal Decomposition', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plots/stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(">> STL分解图已保存: plots/stl_decomposition.png")

# -----
# 主程序
# -----
if __name__ == '__main__':
    # 1. 加载数据
    data_path = '.data/2024_smd_hourly.xlsx'
    df = load_data(data_path)

    print("\n=== 数据概览 ===")
    print(f"形状: {df.shape}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"列名: {list(df.columns)}")
    print(df.head())

    # 2. 画负荷曲线
    plot_load_curve(df)

    # 3. 时序切分（例如用 7月1日 作为切分点）
    # 注意：X 只应包含特征列（不含 target），这里用空 DataFrame 仅取索引对齐
    split_date = pd.Timestamp('2024-07-01')
    X = pd.DataFrame(index=df.index)
    y = df['RT_Demand']
    X_train, X_test, y_train, y_test = train_test_split_ts(X, y, split_date)

    # 4. 画切分示意图
    plot_split_diagram(X_train, X_test, y, split_date)

    # 5. STL 季节分解
    plot_stl_decomposition(df, column='RT_Demand', period=24 * 7 + 1)

    print("\n=== D1 全部完成 ===")
