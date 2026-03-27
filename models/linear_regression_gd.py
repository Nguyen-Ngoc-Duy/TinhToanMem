import numpy as np

def add_bias(X):
    N = X.shape[0]
    return np.hstack([np.ones((N, 1)), X])

def init_weights(X):
    m = X.shape[1]
    return np.zeros(m)

def predict(X, w):
    return X @ w

#hàm mất mát
def compute_loss(X, y, w):
    y_hat = predict(X, w)
    return np.mean((y_hat - y) ** 2)

#GD
def gradient_descent(X, y, w, lr=0.001, epochs=15000):
    N = len(y)
    
    loss_history = []

    for epoch in range(epochs):
        y_hat = predict(X, w)

        gradient = (2/N) * X.T @ (y_hat - y)

        # cập nhật trọng số
        w = w - lr * gradient

        loss = compute_loss(X, y, w)
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, loss_history