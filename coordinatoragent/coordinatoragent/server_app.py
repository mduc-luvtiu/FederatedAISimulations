"""CoordinatorAgent: A Flower / sklearn app (Server)."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from coordinatoragent.strategy import CoordinatorAgentStrategy
from coordinatoragent.task import get_model, get_model_params, set_initial_params
from flwr.server import server 


def server_fn(context: Context) -> ServerAppComponents:
    # Read configuration from run_config
    num_rounds = context.run_config.get("num_rounds", 3)
    penalty = context.run_config.get("penalty", "l2")
    local_epochs = context.run_config.get("local_epochs", 1)

    # Logging for debugging
    print(f"[Server] Starting Federated Learning with:")
    print(f" - Num rounds   : {num_rounds}")
    print(f" - Penalty      : {penalty}")
    print(f" - Local Epochs : {local_epochs}")

    # Create LogisticRegression model
    model = get_model(penalty=penalty, local_epochs=local_epochs)

    # Initialize model parameters
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy (CoordinatorAgent as the "server agent")
    strategy = CoordinatorAgentStrategy(
        fraction_fit=1.0,              # sample 3 clients each round
        fraction_evaluate=1.0,         # all selected clients evaluate
        min_available_clients=2,       # need at least 2 clients to start
        initial_parameters=initial_parameters,
    )

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp instance
app = ServerApp(server_fn=server_fn)


