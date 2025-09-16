import flwr as fl
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def evaluate_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average of the metrics."""
    # Multiply accuracy of each client by number of examples used
    r2_scores = [num_examples * m["r2_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"r2_score": sum(r2_scores) / sum(examples)}


def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Use the entire test set for evaluation."""
        # Tạo dữ liệu test giống với client để đánh giá
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tạo mô hình và thiết lập tham số
        model = LinearRegression()
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1][0]
        
        # Đánh giá
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Server-side evaluation - Round {server_round}: MSE = {mse:.4f}, R = {r2:.4f}")
        return mse, {"r2_score": r2}
    
    return evaluate


# Tạo strategy với các hàm tùy chỉnh
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_round,
    on_evaluate_config_fn=evaluate_round,
    evaluate_fn=get_evaluate_fn(),
)

# Khởi động server
fl.server.start_server(
    server_address="0.0.0.0:8080", 
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
