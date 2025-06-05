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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fl.common.logger.configure(identifier="server")

# Version check
SCRIPT_VERSION = "2025-05-09-v3"
logger.info(f"Running server.py version: {SCRIPT_VERSION}")

# Global variables
global_weights = None
personalized_models = {}
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}
momentum = 0.5  # Reduced momentum for stability
velocity = None

def build_model(input_dim=6):
    """Build an enhanced neural network model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
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
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def pretrain_model():
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
    model.fit(X_scaled, y, epochs=15, batch_size=64, verbose=0)
    logger.info("Pre-training completed.")
    return model.get_weights()

class FeSEM(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters, min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.clients_met = False
        self.client_metrics = {}  # Store client metrics for adaptive weighting

    def initialize_parameters(self, client_manager):
        """Initialize global and personalized model parameters."""
        global global_weights, velocity, personalized_models
        logger.info("Initializing parameters...")
        global_weights = fl.common.parameters_to_ndarrays(self.initial_parameters)
        velocity = [np.zeros_like(w) for w in global_weights]
        personalized_models = {}
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure clients for training with personalized models."""
        logger.info(f"Round {server_round}: Configuring fit...")
        timeout = 300
        start_time = time.time()
        while time.time() - start_time < timeout:
            available_clients = client_manager.num_available()
            logger.info(f"Round {server_round}: Waiting for {self.min_available_clients} clients, {available_clients} available, {time.time() - start_time:.1f}/{timeout}s elapsed")
            if available_clients >= self.min_available_clients:
                logger.info(f"Round {server_round}: {available_clients} clients connected")
                self.clients_met = True
                break
            time.sleep(2)
        if not self.clients_met:
            logger.warning(f"Round {server_round}: Timeout, only {available_clients} clients available")
            return []

        sample_clients = client_manager.sample(
            num_clients=min(self.min_fit_clients, available_clients),
            min_num_clients=self.min_fit_clients
        )
        logger.info(f"Round {server_round}: Sampled {len(sample_clients)} clients for fit: {[str(client.cid) for client in sample_clients]}")

        fit_ins = []
        for client in sample_clients:
            cid = str(client.cid)
            if cid not in personalized_models:
                personalized_models[cid] = [w.copy() for w in global_weights]
            client_weights = personalized_models[cid]
            fit_ins.append((client, fl.common.FitIns(fl.common.ndarrays_to_parameters(client_weights), {"round": server_round})))
        return fit_ins

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates with adaptive weighting based on accuracy and data size."""
        global global_weights, velocity, personalized_models
        if not results:
            logger.warning(f"Round {server_round}: No results received. Failures: {len(failures)}")
            return None, {}

        # Compute adaptive weights based on accuracy and data size
        total_score = 0
        client_scores = {}
        total_samples = sum(res.num_examples for _, res in results)
        for client, res in results:
            cid = str(client.cid)
            accuracy = self.client_metrics.get(cid, {}).get("accuracy", 0.5)
            data_weight = res.num_examples / total_samples if total_samples > 0 else 1.0
            score = (0.7 * accuracy + 0.3 * data_weight)  # Combine accuracy (70%) and data size (30%)
            client_scores[cid] = score
            total_score += score

        if total_score == 0:
            total_score = 1.0  # Avoid division by zero
        client_weights = {cid: score / total_score for cid, score in client_scores.items()}

        # Aggregate updates for the global model
        aggregated_updates = [np.zeros_like(w) for w in global_weights]
        for client, res in results:
            cid = str(client.cid)
            client_update = fl.common.parameters_to_ndarrays(res.parameters)
            weight = client_weights.get(cid, 1.0 / len(results))
            for i in range(len(client_update)):
                aggregated_updates[i] += client_update[i] * weight

        # Apply momentum to the global update
        update = [aggregated_updates[i] - global_weights[i] for i in range(len(global_weights))]
        velocity = [momentum * v + u for v, u in zip(velocity, update)]
        global_weights = [w + v for w, v in zip(global_weights, velocity)]

        # Update personalized models
        for client, res in results:
            cid = str(client.cid)
            local_weights = fl.common.parameters_to_ndarrays(res.parameters)
            personalized_models[cid] = [
                0.6 * lw + 0.4 * gw for lw, gw in zip(local_weights, global_weights)
            ]

        np.save("C:/Users/rkjra/Desktop/FL/FeSEM/final_weights.npy", np.array(global_weights, dtype=object), allow_pickle=True)
        return fl.common.ndarrays_to_parameters(global_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure clients for evaluation using personalized models."""
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

        sample_clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients
        )
        logger.info(f"Round {server_round}: Sampled {len(sample_clients)} clients for evaluation: {[str(client.cid) for client in sample_clients]}")

        eval_ins = []
        for client in sample_clients:
            cid = str(client.cid)
            client_weights = personalized_models.get(cid, global_weights)
            eval_ins.append((client, fl.common.EvaluateIns(fl.common.ndarrays_to_parameters(client_weights), {})))
        return eval_ins

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics from clients and store metrics."""
        global metrics_history
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results received. Failures: {len(failures)}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        total_loss, total_accuracy, total_f1, total_samples = 0, 0, 0, 0
        for client, res in results:
            cid = str(client.cid)
            total_loss += res.loss * res.num_examples
            total_accuracy += res.metrics["accuracy"] * res.num_examples
            total_f1 += res.metrics["f1_score"] * res.num_examples
            total_samples += res.num_examples
            self.client_metrics[cid] = res.metrics

        if total_samples == 0:
            logger.warning(f"Round {server_round}: No valid evaluation results")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_f1 = total_f1 / total_samples
        logger.info(f"Round {server_round} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}")

        metrics_history["loss"].append((server_round, avg_loss))
        metrics_history["accuracy"].append((server_round, avg_accuracy))
        metrics_history["f1_score"].append((server_round, avg_f1))
        np.save("C:/Users/rkjra/Desktop/FL/FeSEM/metrics_history.npy", metrics_history, allow_pickle=True)
        return avg_loss, {"accuracy": avg_accuracy, "f1_score": avg_f1}

    def evaluate(self, server_round, parameters):
        """Evaluate the global model on the server side."""
        logger.info(f"Server evaluate called for round {server_round}...")
        try:
            X_test = np.load("C:/Users/rkjra/Desktop/FL/FeSEM/X_test.npy")
            y_test = np.load("C:/Users/rkjra/Desktop/FL/FeSEM/y_test.npy")
            logger.info(f"Test set class distribution: {dict(pd.Series(y_test).value_counts())}")

            model = build_model(input_dim=X_test.shape[1])
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
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
    """Start the federated learning server."""
    initial_weights = pretrain_model()
    strategy = FeSEM(
        initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )
    server_address = "127.0.0.1:9000"
    logger.info(f"🚀 Starting server at: {server_address}")
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
    np.save("C:/Users/rkjra/Desktop/FL/FeSEM/metrics_history.npy", metrics_history, allow_pickle=True)

if __name__ == "__main__":
    run_server()
