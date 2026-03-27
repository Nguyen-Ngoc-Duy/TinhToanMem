import numpy as np

# Z-score
def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    return (X - mean) / std, mean, std

#chia tập dữ liệu để train/test
def train_test_split(X, y, test_size=0.2):
    N = len(y)
    idx = int(N * (1 - test_size))

    return X[:idx], X[idx:], y[:idx], y[idx:]