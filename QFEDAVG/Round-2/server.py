import flwr as fl
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import time
from sklearn.metrics import f1_score

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
fl.common.logger.configure(identifier="server")

# Version check
SCRIPT_VERSION = "2025-04-25-v12"
logger.info(f"Running server.py version: {SCRIPT_VERSION}")

# Global variables to store metrics and weights
final_weights = None
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}
test_metrics_history = {"test_loss": [], "test_accuracy": [], "test_f1_score": []}

def build_model():
    """Build a simplified neural network model with increased regularization."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(6,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    return model

class QFedAvg(fl.server.strategy.FedAvg):
    """Custom q-FedAvg strategy with FedProx for non-IID data."""
    def __init__(self, q=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.initial_clients_met = False

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training, wait for initial clients."""
        if not self.initial_clients_met and server_round == 1:
            timeout = 60
            start_time = time.time()
            while time.time() - start_time < timeout:
                available_clients = client_manager.num_available()
                logger.info(f"Round {server_round}: Waiting for 3 clients, {available_clients} available, {time.time() - start_time:.1f}/{timeout}s elapsed")
                if available_clients >= self.min_available_clients:
                    logger.info(f"Round {server_round}: All {available_clients} clients connected")
                    self.initial_clients_met = True
                    break
                time.sleep(2)
            if not self.initial_clients_met:
                logger.warning(f"Round {server_round}: Timeout, only {available_clients} clients available")
                return []
        if not self.initial_clients_met:
            available_clients = client_manager.num_available()
            logger.info(f"Round {server_round}: Checking initial clients - {available_clients} available")
            return []
        config = {"round": server_round, "mu": 0.1}
        available_clients = client_manager.num_available()
        logger.info(f"Round {server_round}: {available_clients} clients available")
        sample_clients = client_manager.sample(
            num_clients=min(self.min_available_clients, available_clients),
            min_num_clients=1
        )
        logger.info(f"Configured fit for round {server_round} with {len(sample_clients)} clients sampled")
        return [(client, fl.common.FitIns(parameters, config)) for client in sample_clients]

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client weights using q-FedAvg."""
        global final_weights
        if not results:
            logger.warning(f"No results received in round {server_round}. Failures: {len(failures)}")
            return None, {}
        logger.info(f"Received {len(results)} results and {len(failures)} failures in round {server_round}")
        aggregated_weights = []
        total_samples = 0
        for _, fit_res in results:
            logger.debug(f"Processing fit result with parameters type: {type(fit_res.parameters)}")
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            logger.debug(f"Converted weights to {len(weights)} ndarrays")
            num_examples = fit_res.num_examples
            if not aggregated_weights:
                aggregated_weights = [w * (num_examples ** self.q) for w in weights]
            else:
                for i in range(len(weights)):
                    aggregated_weights[i] += weights[i] * (num_examples ** self.q)
            total_samples += (num_examples ** self.q)
        aggregated_weights = [w / total_samples for w in aggregated_weights]
        final_weights = aggregated_weights
        logger.info(f"Aggregated weights for round {server_round}, total samples: {total_samples}")
        np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/final_weights.npy", np.array(final_weights, dtype=object), allow_pickle=True)
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure clients for evaluation."""
        config = {"round": server_round}
        available_clients = client_manager.num_available()
        sample_clients = client_manager.sample(
            num_clients=min(self.min_available_clients, available_clients),
            min_num_clients=1
        )
        logger.info(f"Configured evaluate for round {server_round} with {len(sample_clients)} clients")
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in sample_clients]

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics from clients."""
        global metrics_history
        if not results:
            logger.warning(f"No evaluation results received in round {server_round}. Failures: {len(failures)}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}
        total_loss, total_accuracy, total_f1, total_samples = 0, 0, 0, 0
        for _, evaluate_res in results:
            total_loss += evaluate_res.loss * evaluate_res.num_examples
            total_accuracy += evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            total_f1 += evaluate_res.metrics["f1_score"] * evaluate_res.num_examples
            total_samples += evaluate_res.num_examples
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
        logger.info(f"Round {server_round} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}")

        metrics_history["loss"].append((server_round, avg_loss))
        metrics_history["accuracy"].append((server_round, avg_accuracy))
        metrics_history["f1_score"].append((server_round, avg_f1))

        np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/metrics_history.npy", metrics_history, allow_pickle=True)
        return avg_loss, {"accuracy": avg_accuracy, "f1_score": avg_f1}

def run_server():
    """Start the federated learning server."""
    initial_weights = build_model().get_weights()
    strategy = QFedAvg(
        q=0.1,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )
    server_address = "127.0.0.1:9000"
    logger.info(f"🚀 Starting server at: {server_address}")
    logger.info("Waiting 60 seconds for clients to connect...")
    time.sleep(60)
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=20, round_timeout=300),
            strategy=strategy
        )
        logger.info("✅ Server finished running.")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        raise
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/metrics_history.npy", metrics_history, allow_pickle=True)
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/test_metrics_history.npy", test_metrics_history, allow_pickle=True)

if __name__ == "__main__":
    run_server()
