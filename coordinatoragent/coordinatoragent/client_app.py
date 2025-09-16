"""CoordinatorAgent: A Flower / sklearn app (Client)."""

import warnings
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from coordinatoragent.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, penalty: str):
        self.model = model
        self.penalty = penalty
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        """Train model with parameters received from server."""
        set_model_params(self.model, parameters)

        # Read training configuration sent by CoordinatorAgentStrategy
        local_epochs = config.get("local_epochs", 1)
        print(f"[Client] Training with local_epochs={local_epochs}")

        # Ignore convergence failure warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Recreate model with new local_epochs but same penalty
            self.model = get_model(self.penalty, local_epochs)
            set_model_params(self.model, parameters)
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate model with parameters from server."""
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        print(f"[Client] Evaluation results - loss: {loss:.4f}, acc: {accuracy:.4f}")

        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    """Create a Flower client instance."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Penalty from run_config, but local_epochs will come from server agent
    penalty = context.run_config.get("penalty", "l2")

    # Create model with dummy local_epochs=1 (will be replaced in fit)
    model = get_model(penalty=penalty, local_epochs=1)
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, penalty).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
