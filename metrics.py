import numpy as np


def _to_numpy(y_true, y_pred):
    """统一转 numpy 数组，并校验长度一致。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 和 y_pred shape 不一致: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred):
    """平均绝对误差，单位与输入一致（MW）。"""
    y_true, y_pred = _to_numpy(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """均方根误差，单位与输入一致（MW）。"""
    y_true, y_pred = _to_numpy(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-9):
    """
    MAPE，返回百分数（不是小数）。
    分母接近 0 的点自动跳过；全被跳过时返回 nan。
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true, y_pred, y_train, season=24):
    """
    MASE = MAE / MAE_naive
    分母 MAE_naive 用训练集算 seasonal naive (lag=season) 的 MAE。
    只接收 y_train，不看测试集 —— 这是硬约束，防止未来信息泄露。
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season:
        return float("nan")
    naive_err = np.abs(y_train[season:] - y_train[:-season])
    naive_mae = float(np.mean(naive_err))
    if naive_mae < 1e-9:
        return float("nan")   # 训练集无变化，MASE 无意义
    return mae(y_true, y_pred) / naive_mae


def all_metrics(y_true, y_pred, y_train, season=24):
    """一次性返回四个指标，回测里直接调用。"""
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, season=season),
    }
