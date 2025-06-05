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
SCRIPT_VERSION = "2025-05-09-v15"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

def build_model(input_dim: int, device_name: str) -> Sequential:
    """Build an enhanced neural network model matching the server's architecture."""
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
    logger.info(f"Client {device_name}: Built model with {len(model.get_weights())} weight arrays")
    return model

def lr_schedule(epoch: int) -> float:
    """Learning rate decay schedule."""
    initial_lr = 0.002
    decay = 0.05
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    return lr

def balance_data_with_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balance the dataset using SMOTE, adjusting k_neighbors dynamically."""
    class_dist = pd.Series(y).value_counts()
    logger.info(f"Class distribution before SMOTE: {dict(class_dist)}")
    
    min_samples = class_dist.min()
    if min_samples < 2:
        logger.warning("Some classes have fewer than 2 samples. Skipping SMOTE.")
        return X, y

    k_neighbors = min(3, min_samples - 1)
    logger.info(f"Using k_neighbors={k_neighbors} for SMOTE based on smallest class size ({min_samples} samples)")
    
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logger.info(f"Class distribution after SMOTE: {dict(pd.Series(y_balanced).value_counts())}")
        return X_balanced, y_balanced
    except Exception as e:
        logger.error(f"SMOTE failed: {e}. Falling back to original data.")
        return X, y

def run_client(device_name: str, X_dev: np.ndarray, y_dev: np.ndarray) -> None:
    """Run the federated learning client for a specific device."""
    logger.info(f"🖥️ Running client for device: {device_name}")
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution before balancing: {class_dist}")

    X_dev, y_dev = balance_data_with_smote(X_dev, y_dev)

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
            """Train the model on local data."""
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

            lr_scheduler = LearningRateScheduler(lr_schedule)
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
            history = model.fit(
                self.X_labeled, self.y_labeled,
                epochs=5,
                batch_size=64,
                callbacks=[lr_scheduler, early_stopping],
                verbose=0
            )

            logger.info(f"Client {device_name}: Training completed for round {self.round}, loss: {history.history['loss'][-1]:.4f}, accuracy: {history.history['accuracy'][-1]:.4f}")
            return model.get_weights(), len(self.X_labeled), {"loss": float(history.history['loss'][-1])}

        def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
            """Evaluate the model on local data."""
            logger.info(f"Client {device_name}: evaluate called for round {self.round} with config: {config}")
            cluster_id = config.get("cluster_id", 0)
            logger.info(f"Client {device_name}: Evaluating with cluster {cluster_id} model")

            logger.info(f"Client {device_name}: Type of parameters received in evaluate: {type(parameters)}")
            if not isinstance(parameters, list):
                logger.error(f"Client {device_name}: Expected list of weights in evaluate, got {type(parameters)}")
                raise ValueError(f"Expected list of weights in evaluate, got {type(parameters)}")
            global_weights = parameters

            try:
                model.set_weights(global_weights)
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
            for label in range(6):
                mask = self.y_labeled == label
                if mask.sum() > 0:
                    class_acc[label] = accuracy_score(self.y_labeled[mask], y_pred_classes[mask])
            logger.info(f"Client {device_name}: Class-wise accuracy: {class_acc}")

            # Temporarily skip cluster loss computation since all_cluster_weights is not sent
            # all_cluster_weights = config.get("all_cluster_weights", [])
            # logger.info(f"Client {device_name}: Received weights for {len(all_cluster_weights)} clusters")
            # cluster_losses = {}
            # for cluster_idx, cluster_params in enumerate(all_cluster_weights):
            #     try:
            #         cluster_weights = fl.common.parameters_to_ndarrays(cluster_params)
            #         start_time = time.time()
            #         model.set_weights(cluster_weights)
            #         cluster_loss = model.evaluate(self.X_labeled, self.y_labeled, verbose=0)[0]
            #         cluster_eval_time = time.time() - start_time
            #         cluster_losses[f"cluster_loss_{cluster_idx}"] = float(cluster_loss)
            #         logger.info(f"Client {device_name}: Loss for cluster {cluster_idx}: {cluster_loss:.4f}, took {cluster_eval_time:.2f} seconds")
            #     except Exception as e:
            #         logger.error(f"Client {device_name}: Failed to evaluate cluster {cluster_idx}: {e}")
            #         cluster_losses[f"cluster_loss_{cluster_idx}"] = float("inf")

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
            # metrics.update(cluster_losses)
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
