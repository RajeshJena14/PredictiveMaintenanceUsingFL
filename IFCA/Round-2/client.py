import flwr as fl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import argparse
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Configure logging
logger = logging.getLogger("client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Version check
SCRIPT_VERSION = "2025-05-09-v19"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

def build_model(input_dim: int, device_name: str) -> Sequential:
    """Build an enhanced neural network model with device-specific dropout rates."""
    # Reduce dropout rates for Client H to prevent underfitting
    dropout_rates = [0.3, 0.2, 0.2, 0.1] if device_name == 'L' else [0.3, 0.2, 0.2, 0.1]  # Adjusted for H
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rates[0]),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rates[1]),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rates[2]),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rates[3]),
        Dense(6, activation='softmax')
    ])
    logger.info(f"Client {device_name}: Built model with {len(model.get_weights())} weight arrays, dropout rates: {dropout_rates}")
    return model

def lr_schedule(epoch: int, device_name: str) -> float:
    """Learning rate decay schedule, unified decay rate for all clients."""
    initial_lr = 0.002
    decay = 0.02  # Slower decay for all clients, including H
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    return lr

def balance_data_with_smote(X: np.ndarray, y: np.ndarray, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Balance the dataset using SMOTE with device-specific k_neighbors."""
    class_dist = pd.Series(y).value_counts()
    logger.info(f"Client {device_name}: Class distribution before SMOTE: {dict(class_dist)}")
    
    min_samples = class_dist.min()
    if min_samples < 2:
        logger.warning(f"Client {device_name}: Some classes have fewer than 2 samples. Skipping SMOTE.")
        return X, y

    # Increase k_neighbors for Client H to better handle minority classes
    k_neighbors = min(5, min_samples - 1) if device_name == 'L' else min(3, min_samples - 1) if device_name == 'M' else min(7, min_samples - 1)
    logger.info(f"Client {device_name}: Using k_neighbors={k_neighbors} for SMOTE based on smallest class size ({min_samples} samples)")
    
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logger.info(f"Client {device_name}: Class distribution after SMOTE: {dict(pd.Series(y_balanced).value_counts())}")
        return X_balanced, y_balanced
    except Exception as e:
        logger.error(f"Client {device_name}: SMOTE failed: {e}. Falling back to original data.")
        return X, y

def run_client(device_name: str, X_dev: np.ndarray, y_dev: np.ndarray) -> None:
    """Run the federated learning client for a specific device."""
    logger.info(f"🖥️ Running client for device: {device_name}")
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution before balancing: {class_dist}")

    X_dev, y_dev = balance_data_with_smote(X_dev, y_dev, device_name)

    logger.info(f"Client {device_name}: Input feature range - min: {np.min(X_dev):.4f}, max: {np.max(X_dev):.4f}")
    if np.any(np.isnan(X_dev)) or np.any(np.isinf(X_dev)):
        logger.error(f"Client {device_name}: Input data X_dev contains NaN or Inf values.")
        raise ValueError("Input data X_dev contains NaN or Inf values.")
    if np.any(np.isnan(y_dev)) or np.any(np.isinf(y_dev)):
        logger.error(f"Client {device_name}: Labels y_dev contain NaN or Inf values.")
        raise ValueError("Labels y_dev contain NaN or Inf values.")

    model = build_model(X_dev.shape[1], device_name)
    X_labeled, y_labeled = X_dev, y_dev

    logger.info(f"Client {device_name}: Labeled data: {len(X_labeled)} samples")

    class IFCAClient(fl.client.NumPyClient):
        def __init__(self, X_labeled: np.ndarray, y_labeled: np.ndarray):
            self.round = 0
            self.X_labeled = X_labeled
            self.y_labeled = y_labeled
            self.metrics_history = {
                'loss': [],
                'accuracy': [],
                'f1_score': []
            }
            logger.info(f"Client {device_name}: Initialized IFCAClient instance")

        def get_parameters(self, config: dict) -> list[np.ndarray]:
            """Return model parameters."""
            logger.info(f"Client {device_name}: get_parameters called with config: {config}")
            weights = model.get_weights()
            logger.info(f"Client {device_name}: Sending {len(weights)} weight arrays to server")
            return weights

        def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
            """Train the model on local data with device-specific epochs."""
            self.round += 1
            logger.info(f"Client {device_name}: fit called for round {self.round} with config: {config}")
            cluster_id = config.get("cluster_id", 0)
            logger.info(f"Client {device_name}: Assigned to cluster {cluster_id}")

            logger.info(f"Client {device_name}: Type of parameters received in fit: {type(parameters)}")
            if not isinstance(parameters, list):
                logger.error(f"Client {device_name}: Expected list of weights in fit, got {type(parameters)}")
                raise ValueError(f"Expected list of weights in fit, got {type(parameters)}")
            global_weights = parameters

            try:
                model.set_weights(global_weights)
            except Exception as e:
                logger.error(f"Client {device_name}: Failed to set weights in fit: {e}")
                raise

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            num_samples = len(self.X_labeled)
            # Increase epochs for Client H to ensure better convergence
            epochs = 10 if device_name == 'L' and num_samples < 1000 else 8 if device_name == 'H' else 5
            logger.info(f"Client {device_name}: Training for {epochs} epochs due to {num_samples} samples")

            lr_scheduler = LearningRateScheduler(lambda epoch: lr_schedule(epoch, device_name))
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
            history = model.fit(
                self.X_labeled, self.y_labeled,
                epochs=epochs,
                batch_size=64,
                callbacks=[lr_scheduler, early_stopping],
                verbose=0
            )

            logger.info(f"Client {device_name}: Training completed for round {self.round}, loss: {history.history['loss'][-1]:.4f}, accuracy: {history.history['accuracy'][-1]:.4f}")
            return model.get_weights(), len(self.X_labeled), {"loss": float(history.history['loss'][-1])}

        def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
            """Evaluate the model on local data and compute losses for all clusters with error handling."""
            logger.info(f"Client {device_name}: evaluate called for round {self.round} with config: {config}")
            cluster_id = config.get("cluster_id", 0)
            logger.info(f"Client {device_name}: Evaluating with cluster {cluster_id} model")

            logger.info(f"Client {device_name}: Type of parameters received in evaluate: {type(parameters)}")
            if not isinstance(parameters, list):
                logger.error(f"Client {device_name}: Expected list of weights in evaluate, got {type(parameters)}")
                raise ValueError(f"Expected list of weights in evaluate, got {type(parameters)}")

            # Extract the flag and weights from parameters
            if len(parameters) < 1:
                logger.error(f"Client {device_name}: Parameters list is too short: {len(parameters)}")
                raise ValueError("Parameters list is too short")

            has_all_weights = bool(parameters[0][0] > 0.5)  # Flag: 1.0 means all weights are included
            logger.info(f"Client {device_name}: Has all_cluster_weights: {has_all_weights}")

            # Extract the cluster weights (skip the flag)
            expected_weights_len = len(model.get_weights())
            cluster_weights = parameters[1:1 + expected_weights_len]
            if len(cluster_weights) != expected_weights_len:
                logger.error(f"Client {device_name}: Expected {expected_weights_len} weights, got {len(cluster_weights)}")
                raise ValueError(f"Expected {expected_weights_len} weights, got {len(cluster_weights)}")

            try:
                model.set_weights(cluster_weights)
            except Exception as e:
                logger.error(f"Client {device_name}: Failed to set weights in evaluate: {e}")
                raise

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            if len(self.X_labeled) == 0 or len(self.y_labeled) == 0:
                logger.warning(f"Client {device_name}: No data for evaluation in round {self.round}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}

            logger.info(f"Client {device_name}: Evaluating on dataset with {len(self.X_labeled)} samples")

            start_time = time.time()
            evaluation_results = model.evaluate(self.X_labeled, self.y_labeled, verbose=0)
            eval_time = time.time() - start_time
            logger.info(f"Client {device_name}: Primary evaluation took {eval_time:.2f} seconds")
            loss, acc = evaluation_results
            y_pred = model.predict(self.X_labeled, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            f1 = f1_score(self.y_labeled, y_pred_classes, average='weighted')

            class_acc = {}
            class_f1 = {}
            for label in range(6):
                mask = self.y_labeled == label
                if mask.sum() > 0:
                    class_acc[label] = accuracy_score(self.y_labeled[mask], y_pred_classes[mask])
                    class_f1[label] = f1_score(self.y_labeled[mask], y_pred_classes[mask], average='weighted')
            logger.info(f"Client {device_name}: Class-wise accuracy: {class_acc}")
            logger.info(f"Client {device_name}: Class-wise F1-score: {class_f1}")

            # Compute loss for each cluster if all_cluster_weights are present
            cluster_losses = {}
            if has_all_weights:
                # Extract all_cluster_weights (after the cluster weights)
                all_cluster_weights_flat = parameters[1 + expected_weights_len:]
                # Group into sets of weights for each cluster
                weights_per_cluster = expected_weights_len
                all_cluster_weights = [
                    all_cluster_weights_flat[i:i + weights_per_cluster]
                    for i in range(0, len(all_cluster_weights_flat), weights_per_cluster)
                ]
                logger.info(f"Client {device_name}: Received {len(all_cluster_weights)} cluster weights sets")

                for cluster_idx, cluster_weights in enumerate(all_cluster_weights):
                    try:
                        if len(cluster_weights) != expected_weights_len:
                            logger.error(f"Client {device_name}: Cluster {cluster_idx} has incorrect number of weights: {len(cluster_weights)}")
                            cluster_losses[f"cluster_loss_{cluster_idx}"] = float("inf")
                            continue

                        start_time = time.time()
                        model.set_weights(cluster_weights)
                        cluster_loss = model.evaluate(self.X_labeled, self.y_labeled, verbose=0)[0]
                        cluster_eval_time = time.time() - start_time
                        cluster_losses[f"cluster_loss_{cluster_idx}"] = float(cluster_loss)
                        logger.info(f"Client {device_name}: Loss for cluster {cluster_idx}: {cluster_loss:.4f}, took {cluster_eval_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Client {device_name}: Failed to evaluate cluster {cluster_idx}: {e}")
                        cluster_losses[f"cluster_loss_{cluster_idx}"] = float("inf")

            logger.info(f"Client {device_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            self.metrics_history['loss'].append((self.round, float(loss)))
            self.metrics_history['accuracy'].append((self.round, float(acc)))
            self.metrics_history['f1_score'].append((self.round, float(f1)))
            metrics_file = f"C:/Users/rkjra/Desktop/FL/IFCA/client_{device_name}_metrics.npy"
            np.save(metrics_file, self.metrics_history)

            metrics = {
                "accuracy": float(acc),
                "f1_score": float(f1),
            }
            metrics.update(cluster_losses)
            logger.info(f"Client {device_name}: Returning evaluation results with metrics: {metrics}")
            return loss, len(self.X_labeled), metrics

    server_address = "127.0.0.1:9000"
    logger.info(f"🔗 Attempting to connect to server at: {server_address}")
    max_retries = 10
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            logger.info(f"Client {device_name}: Connection attempt {attempt + 1}/{max_retries} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            fl.client.start_numpy_client(
                server_address=server_address,
                client=IFCAClient(X_labeled, y_labeled),
                grpc_max_message_length=1024*1024*1024
            )
            logger.info(f"✅ Client {device_name} completed all rounds.")
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Client {device_name}: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ Client {device_name}: Failed after {max_retries} attempts: {e}")
                raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a federated learning client for a specific device.")
    parser.add_argument('--device', type=str, choices=['L', 'M', 'H'], required=True)
    args = parser.parse_args()
    device = args.device

    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    df["Failure_Code"] = df["Failure Type"].astype("category").cat.codes
    df["Device_Type"] = df["Type"].astype("category").cat.codes

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
    X, y = df[features], df['Failure_Code']
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    np.save("C:/Users/rkjra/Desktop/FL/IFCA/X_test.npy", X_scaled_test)
    np.save("C:/Users/rkjra/Desktop/FL/IFCA/y_test.npy", y_test)

    device_map = {'L': 0, 'M': 1, 'H': 2}
    device_code = device_map[device]
    X_dev = X_scaled_train[df_train['Device_Type'] == device_code]
    y_dev = y_train[df_train['Device_Type'] == device_code]
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Main: {device} class distribution: {class_dist}")

    logger.info(f"Starting client for device: {device} with {len(y_dev)} samples")
    run_client(device, X_dev, y_dev)
