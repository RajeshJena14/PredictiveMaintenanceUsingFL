import flwr as fl
import numpy as np
from typing import List, Tuple
import json

# Global lists to store aggregated metrics
aggregated_metrics = {"loss": [], "accuracy": []}

# Custom evaluation function to aggregate metrics
def get_evaluate_fn():
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        return 0.0, {}
    return evaluate

# Custom fit config to pass round number
def fit_config(server_round: int):
    return {"round": server_round}

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,
    evaluate_fn=get_evaluate_fn(),
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=lambda weighted_metrics: {
        "loss": np.mean([m["loss"] for _, m in weighted_metrics]),
        "accuracy": np.mean([m["accuracy"] for _, m in weighted_metrics])
    },
    evaluate_metrics_aggregation_fn=lambda weighted_metrics: {
        "loss": np.mean([m["loss"] for _, m in weighted_metrics]),
        "accuracy": np.mean([m["accuracy"] for _, m in weighted_metrics])
    }
)

# Custom server to log metrics
class CustomServer(fl.server.Server):
    def __init__(self, client_manager, strategy):
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit(self, num_rounds: int, timeout: float = None):
        for round_num in range(1, num_rounds + 1):
            res = super().fit(round_num, timeout)
            # Log aggregated metrics after each round
            fit_metrics = res.metrics_distributed_fit
            eval_metrics = res.metrics_distributed
            if fit_metrics:
                print(f"Debug: fit_metrics = {fit_metrics}")  # Debug output
                # Access the aggregated value directly (index 0) instead of assuming [1]
                loss = fit_metrics.get("loss", [0.0])[0]  # Default to 0.0 if not found
                accuracy = fit_metrics.get("accuracy", [0.0])[0]  # Default to 0.0 if not found
                print(f"Round {round_num}: Loss = {loss}, Accuracy = {accuracy}")
                aggregated_metrics["loss"].append(loss)
                aggregated_metrics["accuracy"].append(accuracy)
        return res

# Save metrics to file after training
def save_metrics():
    with open("aggregated_metrics.json", "w") as f:
        json.dump(aggregated_metrics, f)

# Start the server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    server=CustomServer(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy
    ),
)

# Save metrics after server finishes
save_metrics()
