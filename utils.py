import numpy as np

def reduce_by_factor_10(x, pos):
    return '{:.0f}'.format(x / 10)

def create_dataset(x_data, y_data, window_size):
    x, y = [], []
    for i in range(window_size, len(x_data)):
        x.append(x_data[i-window_size:i+1])
        y.append(y_data[i])
    return np.array(x), np.array(y)

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric_regression(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    #rmse = RMSE(pred, true)
    #mape = MAPE(pred, true)
    #mspe = MSPE(pred, true)

    return mae, mse