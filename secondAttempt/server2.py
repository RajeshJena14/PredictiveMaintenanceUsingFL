import flwr as fl
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
import time
import os

# Set up logging with DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enable Flower's DEBUG logging
fl.common.logger.configure(identifier="server")

# Global variable to hold the final model weights and metrics history
final_weights = None
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}

def build_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(6,)),  # Match client input shape
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    return model

# Custom FedProx strategy
class FedProx(fl.server.strategy.FedProx):
    def __init__(self, initial_parameters, mu=0.01, min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3):
        super().__init__(
            proximal_mu=mu,
            initial_parameters=initial_parameters,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients
        )
        self.current_weights = initial_parameters

    def initialize_parameters(self, client_manager):
        logger.info("Initializing parameters")
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1 and self.current_weights is None:
            parameters_to_send = self.initial_parameters
        else:
            parameters_to_send = self.current_weights
        config = {"mu": self.proximal_mu}
        available_clients = client_manager.num_available()
        logger.info(f"Round {server_round}: {available_clients} clients available")
        if available_clients < self.min_fit_clients:
            logger.warning(f"Not enough clients ({available_clients} < {self.min_fit_clients}). Waiting for all clients...")
            time.sleep(10)  # Wait an additional 10 seconds for clients to connect
            available_clients = client_manager.num_available()
            if available_clients < self.min_fit_clients:
                logger.error(f"Still not enough clients ({available_clients} < {self.min_fit_clients}). Aborting round.")
                raise Exception("Insufficient clients available")
        sample_clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_fit_clients
        )
        logger.info(f"Configured fit for round {server_round} with {len(sample_clients)} clients sampled")
        return [(client, fl.common.FitIns(parameters_to_send, config)) for client in sample_clients]

    def aggregate_fit(self, server_round, results, failures):
        global final_weights
        if not results:
            logger.warning(f"No results received in round {server_round}. Failures: {len(failures)}")
            logger.debug(f"Failures details: {failures}")
            return None, {}
        logger.info(f"Received {len(results)} results and {len(failures)} failures in round {server_round}")
        aggregated_weights = []
        total_samples = 0
        for client, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            if not aggregated_weights:
                aggregated_weights = [w * num_examples for w in weights]
            else:
                for i in range(len(weights)):
                    aggregated_weights[i] += weights[i] * num_examples
            total_samples += num_examples

        aggregated_weights = [w / total_samples for w in aggregated_weights]
        self.current_weights = fl.common.ndarrays_to_parameters(aggregated_weights)
        final_weights = aggregated_weights
        logger.info(f"Aggregated weights for round {server_round}, total samples: {total_samples}")
        np.save("final_weights.npy", np.array(final_weights, dtype=object), allow_pickle=True)
        logger.info(f"Saved final_weights.npy after round {server_round}")
        return self.current_weights, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {}
        available_clients = client_manager.num_available()
        if available_clients < self.min_evaluate_clients:
            logger.warning(f"Not enough clients for evaluation ({available_clients} < {self.min_evaluate_clients}). Waiting...")
            time.sleep(10)  # Wait an additional 10 seconds
            available_clients = client_manager.num_available()
            if available_clients < self.min_evaluate_clients:
                logger.error(f"Still not enough clients for evaluation ({available_clients} < {self.min_evaluate_clients}). Skipping evaluation.")
                return []
        sample_clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_evaluate_clients
        )
        logger.info(f"Configured evaluate for round {server_round} with {len(sample_clients)} clients")
        return [(client, fl.common.EvaluateIns(self.current_weights, config)) for client in sample_clients]

    def aggregate_evaluate(self, server_round, results, failures):
        global metrics_history
        if not results:
            logger.warning(f"No evaluation results received in round {server_round}")
            return None, {}
        
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        total_f1 = 0
        client_metrics = {"loss": {}, "accuracy": {}, "f1_score": {}}
        devices = ['L', 'M', 'H']
        client_idx = 0  # Simple counter to map results to devices (assumes order L, M, H)

        for _, evaluate_res in results:
            dev = devices[client_idx % len(devices)]  # Map result to L, M, H cyclically
            loss = evaluate_res.loss
            num_examples = evaluate_res.num_examples
            accuracy = evaluate_res.metrics["accuracy"]
            f1 = evaluate_res.metrics["f1_score"]

            total_loss += loss * num_examples
            total_samples += num_examples
            total_accuracy += accuracy * num_examples
            total_f1 += f1 * num_examples

            # Store per-client metrics
            if dev not in client_metrics["loss"]:
                client_metrics["loss"][dev] = []
                client_metrics["accuracy"][dev] = []
                client_metrics["f1_score"][dev] = []
            client_metrics["loss"][dev].append((server_round, loss))
            client_metrics["accuracy"][dev].append((server_round, accuracy))
            client_metrics["f1_score"][dev].append((server_round, f1))

            client_idx += 1

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_f1 = total_f1 / total_samples
        logger.info(f"Aggregated evaluation for round {server_round} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}")
        
        # Track aggregated metrics
        metrics_history["loss"].append((server_round, avg_loss))
        metrics_history["accuracy"].append((server_round, avg_accuracy))
        metrics_history["f1_score"].append((server_round, avg_f1))

        # Save client metrics
        try:
            existing_client_metrics = np.load("client_metrics_history.npy", allow_pickle=True).item() if os.path.exists("client_metrics_history.npy") else {}
            for metric in ["loss", "accuracy", "f1_score"]:
                for dev in client_metrics[metric]:
                    if dev not in existing_client_metrics.get(metric, {}):
                        existing_client_metrics.setdefault(metric, {})[dev] = []
                    existing_client_metrics[metric][dev].extend(client_metrics[metric][dev])
            np.save("client_metrics_history.npy", existing_client_metrics, allow_pickle=True)
            logger.info(f"Saved client_metrics_history.npy for round {server_round}")
        except Exception as e:
            logger.error(f"Failed to save client_metrics_history.npy: {e}")

        return avg_loss, {"accuracy": avg_accuracy, "f1_score": avg_f1}

    def evaluate(self, server_round, parameters):
        logger.info(f"Server round {server_round}: Skipping centralized evaluation")
        return None

def run_server(initial_weights):
    try:
        strategy = FedProx(
            initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
            mu=0.01,  # Adjusted for better balance
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3
        )
        server_address = "127.0.0.1:9000"
        logger.info(f"🚀 Starting server at: {server_address}")
        logger.info("Waiting 30 seconds for clients to connect...")
        time.sleep(30)
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=5, round_timeout=300),  # Increased from 120 to 300
            strategy=strategy
        )
        logger.info("✅ Server finished running.")
        
        # Save metrics history
        np.save("metrics_history.npy", metrics_history, allow_pickle=True)
        logger.info("Metrics history saved to 'metrics_history.npy'")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        raise

if __name__ == "__main__":
    initial_weights = build_model().get_weights()
    run_server(initial_weights)

    if final_weights is not None:
        logger.info("Final aggregated weights captured!")
        np.save("final_weights.npy", np.array(final_weights, dtype=object), allow_pickle=True)
        logger.info("Final weights saved to 'final_weights.npy'")
    else:
        logger.error("❌ Final weights not set. Saving initial weights as fallback.")
        np.save("final_weights.npy", np.array(initial_weights, dtype=object), allow_pickle=True)
        logger.info("Initial weights saved to 'final_weights.npy' as fallback")
