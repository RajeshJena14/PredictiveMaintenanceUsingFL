import flwr as fl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import argparse
import grpc
import time
from datetime import datetime
from sklearn.utils import resample

# Configure logging
logger = logging.getLogger("client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Version check
SCRIPT_VERSION = "2025-05-09-v2"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

def build_model(input_dim, device_name):
    """Build a simpler neural network model matching the server's architecture."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='softmax')
    ])
    logger.info(f"Client {device_name}: Built model with {len(model.get_weights())} weight arrays")
    return model

def lr_schedule(epoch):
    """Learning rate decay schedule."""
    initial_lr = 0.0001
    decay = 0.1
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    return lr

def oversample_minority_classes(X, y):
    """Oversample minority classes to balance the dataset."""
    df = pd.DataFrame(X)
    df['label'] = y
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()

    balanced_df = pd.DataFrame()
    for label in class_counts.index:
        class_df = df[df['label'] == label]
        if len(class_df) < max_count:
            class_df = resample(class_df, replace=True, n_samples=max_count, random_state=42)
        balanced_df = pd.concat([balanced_df, class_df])

    X_balanced = balanced_df.drop('label', axis=1).values
    y_balanced = balanced_df['label'].values
    return X_balanced, y_balanced

def run_client(device_name, X_dev, y_dev):
    """Run the federated learning client for a specific device."""
    logger.info(f"🖥️ Running client for device: {device_name}")
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution before balancing: {class_dist}")

    # Oversample minority classes
    X_dev, y_dev = oversample_minority_classes(X_dev, y_dev)
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution after balancing: {class_dist}")

    # Verify data integrity
    logger.info(f"Client {device_name}: Input feature range - min: {np.min(X_dev):.4f}, max: {np.max(X_dev):.4f}")
    if np.any(np.isnan(X_dev)) or np.any(np.isinf(X_dev)):
        logger.error(f"Client {device_name}: Input data X_dev contains NaN or Inf values.")
        raise ValueError("Input data X_dev contains NaN or Inf values.")
    if np.any(np.isnan(y_dev)) or np.any(np.isinf(y_dev)):
        logger.error(f"Client {device_name}: Labels y_dev contain NaN or Inf values.")
        raise ValueError("Labels y_dev contain NaN or Inf values.")

    model = build_model(X_dev.shape[1], device_name)
    X_labeled, y_labeled = X_dev, y_dev  # Fully supervised learning

    logger.info(f"Client {device_name}: Labeled data: {len(X_labeled)} samples")

    class FeSEMClient(fl.client.NumPyClient):
        def __init__(self, X_labeled, y_labeled):
            self.round = 0
            self.X_labeled = X_labeled
            self.y_labeled = y_labeled
            self.metrics_history = {
                'loss': [],
                'accuracy': [],
                'f1_score': []
            }

        def get_parameters(self, config):
            """Return model parameters."""
            logger.info(f"Client {device_name}: get_parameters called with config: {config}")
            weights = model.get_weights()
            if any(np.any(np.isnan(w)) for w in weights):
                logger.error(f"Client {device_name}: Model weights contain NaN values.")
                raise ValueError("Model weights contain NaN values.")
            logger.info(f"Client {device_name}: Sending {len(weights)} weight arrays to server")
            return weights

        def fit(self, parameters, config):
            """Train the model on local data."""
            self.round += 1
            logger.info(f"Client {device_name}: fit called for round {self.round} with config: {config}")
            # Debug: Log the type of parameters received
            logger.info(f"Client {device_name}: Type of parameters received in fit: {type(parameters)}")
            # Ensure parameters is a Parameters object
            if not isinstance(parameters, fl.common.Parameters):
                logger.warning(f"Client {device_name}: Expected Parameters object in fit, got {type(parameters)}, attempting to convert")
                try:
                    parameters = fl.common.ndarrays_to_parameters(parameters)
                except Exception as e:
                    logger.error(f"Client {device_name}: Failed to convert parameters to Parameters object: {e}")
                    raise

            global_weights = fl.common.parameters_to_ndarrays(parameters)
            logger.info(f"Client {device_name}: Received {len(global_weights)} weight arrays from server in fit")
            try:
                model.set_weights(global_weights)
            except ValueError as e:
                logger.error(f"Client {device_name}: Failed to set weights in fit: {e}")
                raise

            # Compute class weights to handle imbalance
            unique, counts = np.unique(self.y_labeled, return_counts=True)
            if len(unique) == 0 or any(c == 0 for c in counts):
                logger.warning(f"Client {device_name}: Invalid class distribution for round {self.round}, using uniform weights")
                class_weights = {i: 1.0 for i in range(6)}
            else:
                total_samples = len(self.y_labeled)
                num_classes = len(unique)
                class_weights = {k: (total_samples / (num_classes * v)) for k, v in zip(unique, counts)}
            logger.info(f"Client {device_name}: Round {self.round} class weights: {class_weights}")

            sample_weights = np.array([class_weights[label] for label in self.y_labeled])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            lr_scheduler = LearningRateScheduler(lr_schedule)
            history = model.fit(
                self.X_labeled, self.y_labeled,
                epochs=20,
                batch_size=32,
                sample_weight=sample_weights,
                callbacks=[lr_scheduler],
                verbose=0
            )

            if not history.history or 'loss' not in history.history:
                logger.error(f"Client {device_name}: Training failed for round {self.round}, history invalid")
                return (model.get_weights(), len(self.X_labeled), {"loss": 0.0})

            if any(np.isnan(history.history['loss'])):
                logger.error(f"Client {device_name}: Training produced NaN loss in round {self.round}")
                return (model.get_weights(), len(self.X_labeled), {"loss": 0.0})

            logger.info(f"Client {device_name}: Training completed for round {self.round}, loss: {history.history['loss'][-1]:.4f}, accuracy: {history.history['accuracy'][-1]:.4f}")
            return (model.get_weights(), len(self.X_labeled), {"loss": float(history.history['loss'][-1])})

        def evaluate(self, parameters, config):
            """Evaluate the model on local data."""
            logger.info(f"Client {device_name}: evaluate called with config: {config}")
            # Debug: Log the type of parameters received
            logger.info(f"Client {device_name}: Type of parameters received in evaluate: {type(parameters)}")
            # Ensure parameters is a Parameters object
            if not isinstance(parameters, fl.common.Parameters):
                logger.warning(f"Client {device_name}: Expected Parameters object in evaluate, got {type(parameters)}, attempting to convert")
                try:
                    parameters = fl.common.ndarrays_to_parameters(parameters)
                except Exception as e:
                    logger.error(f"Client {device_name}: Failed to convert parameters to Parameters object: {e}")
                    raise

            global_weights = fl.common.parameters_to_ndarrays(parameters)
            logger.info(f"Client {device_name}: Received {len(global_weights)} weight arrays from server in evaluate")
            try:
                model.set_weights(global_weights)
            except ValueError as e:
                logger.error(f"Client {device_name}: Failed to set weights in evaluate: {e}")
                raise

            # Compute class weights for evaluation
            unique, counts = np.unique(self.y_labeled, return_counts=True)
            if len(unique) == 0 or any(c == 0 for c in counts):
                logger.warning(f"Client {device_name}: Invalid class distribution for evaluation, using uniform weights")
                class_weights = {i: 1.0 for i in range(6)}
            else:
                total_samples = len(self.y_labeled)
                num_classes = len(unique)
                class_weights = {k: (total_samples / (num_classes * v)) for k, v in zip(unique, counts)}
            logger.info(f"Client {device_name}: Evaluation class weights: {class_weights}")

            sample_weights = np.array([class_weights[label] for label in self.y_labeled])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                weighted_metrics=['accuracy']
            )

            if len(self.X_labeled) == 0 or len(self.y_labeled) == 0:
                logger.warning(f"Client {device_name}: No data for evaluation in round {self.round}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}

            try:
                evaluation_results = model.evaluate(
                    self.X_labeled, self.y_labeled,
                    sample_weight=sample_weights,
                    verbose=0
                )
                loss, unweighted_acc, weighted_acc = evaluation_results
                logger.info(f"Client {device_name}: Evaluation - Loss: {loss:.4f}, Unweighted Accuracy: {unweighted_acc:.4f}, Weighted Accuracy: {weighted_acc:.4f}")

                y_pred = model.predict(self.X_labeled, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                f1 = f1_score(self.y_labeled, y_pred_classes, average='weighted')
                pred_dist = dict(pd.Series(y_pred_classes).value_counts())
                logger.info(f"Client {device_name}: Prediction distribution on evaluation: {pred_dist}")

                # Compute class-wise accuracy
                class_acc = {}
                for label in range(6):
                    mask = self.y_labeled == label
                    if mask.sum() > 0:
                        class_acc[label] = accuracy_score(self.y_labeled[mask], y_pred_classes[mask])
                logger.info(f"Client {device_name}: Class-wise accuracy: {class_acc}")
            except Exception as e:
                logger.error(f"Client {device_name}: Error in evaluation: {e}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}

            logger.info(f"Client {device_name} - Loss: {loss:.4f}, Weighted Accuracy: {weighted_acc:.4f}, F1-Score: {f1:.4f}")
            self.metrics_history['loss'].append((self.round, float(loss)))
            self.metrics_history['accuracy'].append((self.round, float(weighted_acc)))
            self.metrics_history['f1_score'].append((self.round, float(f1)))
            metrics_file = f"C:/Users/rkjra/Desktop/FL/FeSEM/client_{device_name}_metrics.npy"
            np.save(metrics_file, self.metrics_history)
            logger.info(f"Client {device_name}: Saved evaluation metrics to {metrics_file}")
            return loss, len(self.X_labeled), {"accuracy": weighted_acc, "f1_score": f1}

    server_address = "127.0.0.1:9000"
    logger.info(f"🔗 Attempting to connect to server at: {server_address}")
    max_retries = 10
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            logger.info(f"Client {device_name}: Connection attempt {attempt + 1}/{max_retries} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            channel = grpc.insecure_channel(server_address)
            grpc.channel_ready_future(channel).result(timeout=30)
            fl.client.start_numpy_client(
                server_address=server_address,
                client=FeSEMClient(X_labeled, y_labeled)
            )
            channel.close()
            logger.info(f"🔌 gRPC channel closed for {device_name}")
            logger.info(f"✅ Client {device_name} completed.")
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
    np.save("C:/Users/rkjra/Desktop/FL/FeSEM/X_test.npy", X_scaled_test)
    np.save("C:/Users/rkjra/Desktop/FL/FeSEM/y_test.npy", y_test)

    device_map = {'L': 0, 'M': 1, 'H': 2}
    device_code = device_map[device]
    X_dev = X_scaled_train[df_train['Device_Type'] == device_code]
    y_dev = y_train[df_train['Device_Type'] == device_code]
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Main: {device} class distribution: {class_dist}")

    logger.info(f"Starting client for device: {device} with {len(y_dev)} samples")
    run_client(device, X_dev, y_dev)
