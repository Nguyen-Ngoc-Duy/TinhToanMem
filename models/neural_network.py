import numpy as np

#hàm relu
def relu(Z):
    return np.maximum(0, Z)

#đạo hàm relu
def relu_derivative(Z):
    return (Z > 0).astype(float)

def init_params(input_size, hidden_size):
    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2. / hidden_size)

    return W1, W2

#lan truyền thuận
def forward(X, W1, W2):
    Z1 = X @ W1 #f1..f8 
    A1 = relu(Z1)

    Z2 = A1 @ W2
    y_hat = Z2   # y là hàm tuyến tính

    return Z1, A1, y_hat

#hàm mất mát
def compute_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)

#lan truyền ngược
def backward(X, y, Z1, A1, y_hat, W2):
    N = X.shape[0]

    # ngõ ra
    dZ2 = (2/N) * (y_hat - y)
    dW2 = A1.T @ dZ2

    # lớp ẩn
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1

    # hạn chế GD quá lớn
    dW1 = np.clip(dW1, -5, 5)
    dW2 = np.clip(dW2, -5, 5)

    return dW1, dW2


def train_nn(X, y, hidden_size=8, lr=0.001, epochs=15000):
    input_size = X.shape[1]

    W1, W2 = init_params(input_size, hidden_size)

    loss_history = []

    for epoch in range(epochs):
        # Thuận
        Z1, A1, y_hat = forward(X, W1, W2)

        # Mất mát
        loss = compute_loss(y_hat, y)
        loss_history.append(loss)

        if np.isnan(loss):
            print("Lỗi")
            break

        # Ngược
        dW1, dW2 = backward(X, y, Z1, A1, y_hat, W2)

        # Cập nhật trọng số
        W1 -= lr * dW1
        W2 -= lr * dW2

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return W1, W2, loss_history


def predict_nn(X, W1, W2):
    _, _, y_hat = forward(X, W1, W2)
    return y_hat