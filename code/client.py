import flwr as fl
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Tạo dữ liệu regression giả lập
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Linear Regression
model = LinearRegression()

# Chia dữ liệu cho từng client
client_id = int(sys.argv[1])
num_samples = len(X_train) // 2
if client_id == 1:
    X_train_client = X_train[:num_samples]
    y_train_client = y_train[:num_samples]
else:
    X_train_client = X_train[num_samples:]
    y_train_client = y_train[num_samples:]


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Trả về các tham số của mô hình (coef_ và intercept_)
        if hasattr(model, 'coef_'):
            return [model.coef_, np.array([model.intercept_])]
        else:
            # Nếu mô hình chưa được huấn luyện, trả về 0
            return [0, np.array([0.0])]

    def set_parameters(self, parameters):
        # Thiết lập tham số cho mô hình
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1][0]

    def fit(self, parameters, config):
        # Thiết lập tham số nhận được từ server
        self.set_parameters(parameters)
        
        # Huấn luyện mô hình trên dữ liệu local
        model.fit(X_train_client, y_train_client)
        
        # Trả về tham số đã được cập nhật
        return self.get_parameters(config), len(X_train_client), {}

    def evaluate(self, parameters, config):
        # Thiết lập tham số nhận được từ server
        self.set_parameters(parameters)
        
        # Đánh giá mô hình trên test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mse, len(X_test), {"r2_score": r2}

# Khởi động Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
