from preprocessing.preprocess import load_data, describe_data, preprocess_data
from normalization.scaler import z_score, train_test_split
from models.linear_regression_gd import *
from evaluation.metrics import *
from models.neural_network import train_nn, predict_nn
import config
import matplotlib.pyplot as plt

df = load_data("data/BIKE DETAILS.csv")
describe_data(df)

df = preprocess_data(df)

X = df[config.FEATURES].values
y = df[config.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, config.TEST_SIZE
)

X_train, mean_X, std_X = z_score(X_train)
X_test = (X_test - mean_X) / std_X

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train, mean_y, std_y = z_score(y_train)
y_test_scaled = (y_test - mean_y) / std_y

X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)

w = init_weights(X_train_bias)

w, loss_history = gradient_descent(
    X_train_bias, y_train.flatten(), w,
    lr=config.LR,
    epochs=config.EPOCHS
)

W1, W2, loss_nn = train_nn(
    X_train, y_train,  
    hidden_size=config.HIDDEN_SIZE,
    lr=config.LR,
    epochs=config.EPOCHS
)

y_pred = predict(X_test_bias, w).reshape(-1, 1)
y_pred_nn = predict_nn(X_test, W1, W2)
y_pred = y_pred * std_y + mean_y
y_pred_nn = y_pred_nn * std_y + mean_y
y_test_real = y_test 

#Sau khi chuẩn hóa bằng z-score
print("===== DATA SAU Z-SCORE (5 dòng đầu) =====")
print(X_train[:5])
print("\n===== MEAN =====")
print(mean_X)
print("\n===== STD =====")
print(std_X)
print("\n===== KIỂM TRA =====")
print("Mean sau scale:", np.mean(X_train, axis=0))
print("Std sau scale:", np.std(X_train, axis=0))


print("\n===== LINEAR REGRESSION =====")
print("MSE:", mse(y_test_real, y_pred))
print("RMSE:", rmse(y_test_real, y_pred))
print("R2:", r2_score(y_test_real, y_pred))

print("\n===== NEURAL NETWORK =====")
print("MSE:", mse(y_test_real, y_pred_nn))
print("RMSE:", rmse(y_test_real, y_pred_nn))
print("R2:", r2_score(y_test_real, y_pred_nn))

plt.figure()
plt.plot(loss_history, label="Linear Regression")
plt.title("Loss - Linear Regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(loss_nn, label="Neural Network")
plt.title("Loss - Neural Network")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(loss_history, label="Linear Regression")
plt.plot(loss_nn, label="Neural Network")
plt.title("So sánh Loss giữa 2 mô hình")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.scatter(y_test, y_pred)
a, b = np.polyfit(y_test.flatten(), y_pred.flatten(), 1)
x_line = np.linspace(min(y_test), max(y_test), 100)
y_line = a * x_line + b
plt.plot(x_line, y_line)
plt.xlabel("Giá trị thật (Real)")
plt.ylabel("Giá trị dự đoán (Prediction)")
plt.title("Đường hồi quy dự đoán")
plt.show()