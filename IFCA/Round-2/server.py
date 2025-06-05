import flwr as fl
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from typing import List, Dict, Tuple, Optional, Union
import grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fl.common.logger.configure(identifier="server")

# Version check
SCRIPT_VERSION = "2025-05-09-v16"
logger.info(f"Running server.py version: {SCRIPT_VERSION}")

# Global variables
NUM_CLUSTERS = 3
cluster_models: List[List[np.ndarray]] = None
client_clusters: Dict[str, int] = {}
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}

def build_model(input_dim: int = 6) -> Sequential:
    """Build an enhanced neural network model."""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(6, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def pretrain_model() -> List[np.ndarray]:
    """Pre-train a base model on a subset of data."""
    logger.info("Pre-training base model...")
    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    df["Failure_Code"] = df["Failure Type"].astype("category").cat.codes
    df["Device_Type"] = df["Type"].astype("category").cat.codes

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
    X, y = df[features], df['Failure_Code']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = build_model(input_dim=X_scaled.shape[1])
    model.fit(X_scaled, y, epochs=20, batch_size=64, verbose=0)
    logger.info("Pre-training completed.")
    return model.get_weights()

class IFCAStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        initial_parameters: fl.common.typing.Parameters,
        num_clusters: int = NUM_CLUSTERS,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3
    ):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.num_clusters = num_clusters
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[fl.common.typing.Parameters]:
        """Initialize cluster models with pre-trained weights."""
        global cluster_models, client_clusters
        logger.info("Initializing parameters...")
        base_weights = fl.common.parameters_to_ndarrays(self.initial_parameters)
        cluster_models = [[w.copy() for w in base_weights] for _ in range(self.num_clusters)]
        client_clusters = {}
        logger.info(f"Initialized {self.num_clusters} cluster models.")
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitIns]]:
        """Configure clients for training with cluster-specific models."""
        logger.info(f"Round {server_round}: Configuring fit...")
        timeout = 300
        start_time = time.time()
        while time.time() - start_time < timeout:
            available_clients = client_manager.num_available()
            logger.info(f"Round {server_round}: Waiting for {self.min_available_clients} clients, {available_clients} available, {time.time() - start_time:.1f}/{timeout}s elapsed")
            if available_clients >= self.min_available_clients:
                logger.info(f"Round {server_round}: {available_clients} clients connected")
                break
            time.sleep(2)
        if available_clients < self.min_available_clients:
            logger.warning(f"Round {server_round}: Timeout, only {available_clients} clients available")
            return []

        sample_clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients
        )
        logger.info(f"Round {server_round}: Sampled {len(sample_clients)} clients for fit: {[str(client.cid) for client in sample_clients]}")

        fit_ins = []
        for client in sample_clients:
            cid = str(client.cid)
            if cid not in client_clusters:
                client_clusters[cid] = np.random.randint(0, self.num_clusters)
            cluster_id = client_clusters[cid]
            cluster_weights = cluster_models[cluster_id]
            parameters = fl.common.ndarrays_to_parameters(cluster_weights)
            config = {"round": server_round, "cluster_id": cluster_id}
            logger.info(f"Round {server_round}: Sending fit instructions to client {cid} for cluster {cluster_id}")
            fit_ins.append((client, fl.common.FitIns(parameters, config)))
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]]
    ) -> Tuple[Optional[fl.common.typing.Parameters], Dict[str, float]]:
        """Aggregate client updates within each cluster with normalized weights."""
        global cluster_models
        if not results:
            logger.warning(f"Round {server_round}: No fit results received. Failures: {len(failures)}")
            return None, {}

        cluster_updates: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(self.num_clusters)]
        for client, res in results:
            cid = str(client.cid)
            cluster_id = client_clusters[cid]
            weights = fl.common.parameters_to_ndarrays(res.parameters)
            cluster_updates[cluster_id].append((weights, res.num_examples))
            logger.info(f"Round {server_round}: Received fit results from client {cid} for cluster {cluster_id}, num_examples: {res.num_examples}")

        for cluster_id in range(self.num_clusters):
            if not cluster_updates[cluster_id]:
                logger.info(f"Round {server_round}: No updates for cluster {cluster_id}")
                continue
            updates, num_examples = zip(*cluster_updates[cluster_id])
            total_samples = sum(num_examples)
            if total_samples == 0:
                logger.warning(f"Round {server_round}: Cluster {cluster_id} has no samples")
                continue
            normalized_weights = np.array([n / total_samples for n in num_examples])
            aggregated_weights = []
            for layer in zip(*updates):
                layer_updates = np.array(layer)
                weights = normalized_weights.reshape(-1, *[1] * (len(layer_updates.shape) - 1))
                weighted_layer = np.sum(layer_updates * weights, axis=0)
                aggregated_weights.append(weighted_layer)
            cluster_models[cluster_id] = aggregated_weights
            logger.info(f"Round {server_round}: Aggregated weights for cluster {cluster_id}")

        np.save("C:/Users/rkjra/Desktop/FL/IFCA/cluster_models.npy", np.array(cluster_models, dtype=object), allow_pickle=True)
        return fl.common.ndarrays_to_parameters(cluster_models[0]), {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.EvaluateIns]]:
        """Configure clients for evaluation using cluster-specific models and send all cluster weights periodically."""
        logger.info(f"Round {server_round}: Configuring evaluate...")
        timeout = 300
        start_time = time.time()
        while time.time() - start_time < timeout:
            available_clients = client_manager.num_available()
            logger.info(f"Round {server_round}: Waiting for {self.min_evaluate_clients} clients for evaluation, {available_clients} available, {time.time() - start_time:.1f}/{timeout}s elapsed")
            if available_clients >= self.min_evaluate_clients:
                logger.info(f"Round {server_round}: {available_clients} clients available for evaluation")
                break
            time.sleep(2)
        if available_clients < self.min_evaluate_clients:
            logger.warning(f"Round {server_round}: Timeout, only {available_clients} clients available for evaluation")
            return []

        available_clients = client_manager.num_available()
        logger.info(f"Round {server_round}: Double-checking: {available_clients} clients still available before sampling")
        if available_clients < self.min_evaluate_clients:
            logger.warning(f"Round {server_round}: Clients dropped, only {available_clients} available")
            return []

        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            sample_clients = client_manager.sample(
                num_clients=self.min_evaluate_clients,
                min_num_clients=self.min_evaluate_clients
            )
            logger.info(f"Round {server_round}: Attempt {attempt + 1}/{max_retries} - Sampled {len(sample_clients)} clients for evaluation: {[str(client.cid) for client in sample_clients]}")

            eval_ins = []
            # Send all_cluster_weights every 3 rounds instead of 5
            send_all_weights = server_round % 3 == 0
            if send_all_weights:
                # Combine cluster weights and all cluster weights into parameters
                all_cluster_weights = [fl.common.ndarrays_to_parameters(weights) for weights in cluster_models]
            else:
                all_cluster_weights = []

            for client in sample_clients:
                cid = str(client.cid)
                cluster_id = client_clusters.get(cid, 0)
                cluster_weights = cluster_models[cluster_id]
                # If sending all weights, prepend a flag and all_cluster_weights to parameters
                if send_all_weights:
                    parameters_to_send = fl.common.ndarrays_to_parameters(
                        [np.array([1.0])] + cluster_weights + [w.tensors for w in all_cluster_weights]
                    )
                else:
                    parameters_to_send = fl.common.ndarrays_to_parameters(
                        [np.array([0.0])] + cluster_weights
                    )
                # Config should only contain serializable types
                config = {
                    "round": server_round,
                    "cluster_id": cluster_id
                }
                logger.info(f"Round {server_round}: Sending evaluate instructions to client {cid} for cluster {cluster_id}, all_cluster_weights included: {send_all_weights}")
                eval_ins.append((client, fl.common.EvaluateIns(parameters_to_send, config)))

            if eval_ins:
                return eval_ins
            logger.warning(f"Round {server_round}: Attempt {attempt + 1}/{max_retries} failed to configure evaluation")
            if attempt < max_retries - 1:
                logger.info(f"Round {server_round}: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        logger.error(f"Round {server_round}: Failed to configure evaluation after {max_retries} attempts")
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.EvaluateRes], Exception]]
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics and reassign clients to clusters with robust failure handling."""
        global metrics_history, client_clusters
        logger.info(f"Round {server_round}: Aggregate evaluate received {len(results)} results and {len(failures)} failures")

        # Handle failures, which may include Exception objects
        for failure in failures:
            if isinstance(failure, tuple) and len(failure) == 2:
                client, res = failure
                try:
                    cid = str(client.cid)
                    logger.error(f"Round {server_round}: Evaluation failed for client {cid}. Failure details: {res}")
                except Exception as e:
                    logger.error(f"Round {server_round}: Could not retrieve failure details: {e}")
            else:
                # Handle case where failure is an Exception object
                logger.error(f"Round {server_round}: Evaluation failure (Exception): {str(failure)}")

        if not results:
            logger.warning(f"Round {server_round}: No evaluation results received. Failures: {len(failures)}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        # Reassign clients to the best cluster if all_cluster_weights were sent
        send_all_weights = server_round % 3 == 0
        if send_all_weights:
            for client, res in results:
                cid = str(client.cid)
                client_losses = {}
                for cluster_id in range(self.num_clusters):
                    loss_key = f"cluster_loss_{cluster_id}"
                    loss = res.metrics.get(loss_key, float("inf"))
                    client_losses[cluster_id] = loss
                best_cluster = min(client_losses, key=client_losses.get)
                # Only reassign if the loss is below a threshold to avoid bad assignments
                loss_threshold = 1.0  # Adjust based on observed loss values
                if client_losses[best_cluster] < loss_threshold:
                    old_cluster = client_clusters.get(cid, 0)
                    client_clusters[cid] = best_cluster
                    logger.info(f"Round {server_round}: Client {cid} reassigned from cluster {old_cluster} to cluster {best_cluster} with loss {client_losses[best_cluster]:.4f}")
                else:
                    logger.info(f"Round {server_round}: Client {cid} not reassigned; best cluster loss {client_losses[best_cluster]:.4f} exceeds threshold {loss_threshold}")

        total_loss, total_accuracy, total_f1, total_samples = 0.0, 0.0, 0.0, 0
        for client, res in results:
            total_loss += res.loss * res.num_examples
            total_accuracy += res.metrics["accuracy"] * res.num_examples
            total_f1 += res.metrics["f1_score"] * res.num_examples
            total_samples += res.num_examples
            logger.info(f"Round {server_round}: Client {client.cid} evaluation - Loss: {res.loss:.4f}, Accuracy: {res.metrics['accuracy']:.4f}, F1-Score: {res.metrics['f1_score']:.4f}")

        if total_samples == 0:
            logger.warning(f"Round {server_round}: No valid evaluation results")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_f1 = total_f1 / total_samples
        logger.info(f"Round {server_round} - Aggregated Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}")

        metrics_history["loss"].append((server_round, avg_loss))
        metrics_history["accuracy"].append((server_round, avg_accuracy))
        metrics_history["f1_score"].append((server_round, avg_f1))
        np.save("C:/Users/rkjra/Desktop/FL/IFCA/metrics_history.npy", metrics_history, allow_pickle=True)
        return avg_loss, {"accuracy": avg_accuracy, "f1_score": avg_f1}

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Evaluate the global model (average of cluster models) on the server side."""
        logger.info(f"Server evaluate called for round {server_round}...")
        try:
            X_test = np.load("C:/Users/rkjra/Desktop/FL/IFCA/X_test.npy")
            y_test = np.load("C:/Users/rkjra/Desktop/FL/IFCA/y_test.npy")
            logger.info(f"Test set class distribution: {dict(pd.Series(y_test).value_counts())}")

            global_weights = []
            for layer in zip(*cluster_models):
                layer_weights = np.mean(np.array(layer), axis=0)
                global_weights.append(layer_weights)

            model = build_model(input_dim=X_test.shape[1])
            model.set_weights(global_weights)
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            pred_dist = dict(pd.Series(y_pred_classes).value_counts())
            logger.info(f"Server prediction distribution: {pred_dist}")
            logger.info(f"Server evaluation in round {server_round} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            return loss, {"accuracy": acc, "f1_score": f1}
        except Exception as e:
            logger.error(f"Server evaluation failed in round {server_round}: {e}")
            return None, {}

def run_server():
    """Start the federated learning server with IFCA strategy."""
    initial_weights = pretrain_model()
    strategy = IFCAStrategy(
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        num_clusters=NUM_CLUSTERS,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )
    server_address = "127.0.0.1:9000"
    logger.info(f"🚀 Starting server at: {server_address}")
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=20, round_timeout=1200),
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024
        )
        logger.info("✅ Server finished running.")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        raise
    np.save("C:/Users/rkjra/Desktop/FL/IFCA/metrics_history.npy", metrics_history, allow_pickle=True)

if __name__ == "__main__":
    run_server()
