import flwr as fl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import argparse
import grpc
import time
from datetime import datetime

# CyclicLR callback definition
class CyclicLR(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=0.001, max_lr=0.01, step_size=1000, mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        return self.base_lr

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1
        lr = self.clr()
        K.set_value(self.model.optimizer.lr, lr)
        logs['lr'] = lr

# Configure logging
logger = logging.getLogger("client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Version check
SCRIPT_VERSION = "2025-04-25-v14"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

def build_model(input_dim):
    """Build a simplified neural network model with increased regularization."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    return model

def run_client(device_name, X_dev, y_dev):
    """Run the federated learning client for a specific device."""
    logger.info(f"🖥️ Running client for device: {device_name}")
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution {class_dist}")
    model = build_model(X_dev.shape[1])

    class QFedAvgClient(fl.client.NumPyClient):
        def __init__(self, X_dev, y_dev):
            self.round = 0
            self.X_dev = X_dev
            self.y_dev = y_dev
            self.metrics_history = {
                'loss': [],
                'accuracy': [],
                'f1_score': []
            }

        def get_parameters(self, config):
            logger.info(f"Client {device_name}: get_parameters called with config: {config}")
            weights = model.get_weights()
            logger.debug(f"Client {device_name}: get_parameters returning weights with {len(weights)} arrays")
            return weights

        def fit(self, parameters, config):
            self.round += 1
            logger.info(f"Client {device_name}: fit called for round {self.round} with config: {config}")
            # Set model weights with received parameters to ensure compatibility
            model.set_weights(parameters)
            # Calculate base_lr and max_lr for CyclicLR
            base_lr = 0.001 * (0.98 ** (self.round - 1))
            max_lr = 10 * base_lr  # Adjust this factor as needed
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=base_lr, clipnorm=1.0
            )
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            # Add FedProx proximal term using aligned weights
            mu = config.get("mu", 0.1)
            global_weights = parameters
            global_weights_tensors = [tf.constant(w) for w in global_weights]
            local_weights = model.get_weights()  # Get weights after setting to ensure alignment
            local_weights_tensors = [tf.constant(w) for w in local_weights]
            # Debug: Log shapes of local and global weights
            logger.debug(f"Client {device_name}: Round {self.round} - Local weights shapes: {[w.shape for w in local_weights]}")
            logger.debug(f"Client {device_name}: Round {self.round} - Global weights shapes: {[w.shape for w in global_weights]}")
            proximal_terms = []
            for w_local, w_global in zip(local_weights_tensors, global_weights_tensors):
                if w_local.shape != w_global.shape:
                    logger.error(f"Client {device_name}: Shape mismatch in round {self.round} - Local weight shape {w_local.shape} vs Global weight shape {w_global.shape}")
                    raise ValueError(f"Shape mismatch: {w_local.shape} vs {w_global.shape}")
                proximal_terms.append(tf.reduce_sum(tf.square(w_local - w_global)))
            total_proximal_term = tf.add_n(proximal_terms) if proximal_terms else tf.constant(0.0)
            model.add_loss(lambda: mu / 2 * total_proximal_term)
            X_train_dev, X_val_dev, y_train_dev, y_val_dev = train_test_split(
                self.X_dev, self.y_dev, test_size=0.1, random_state=42
            )
            if len(X_train_dev) == 0 or len(y_train_dev) == 0:
                logger.error(f"Client {device_name}: Empty training data for round {self.round}")
                return (model.get_weights(), 0, {"loss": 0.0, "val_loss": 0.0})
            unique, counts = np.unique(y_train_dev, return_counts=True)
            if len(unique) == 0 or any(c == 0 for c in counts):
                logger.warning(f"Client {device_name}: Invalid class distribution for round {self.round}, using uniform weights")
                class_weights = {i: 1.0 for i in range(6)}
            else:
                class_weights = {k: (1.0 / v) * (len(y_train_dev) / len(unique)) for k, v in zip(unique, counts)}
            logger.info(f"Client {device_name}: Round {self.round} class weights: {class_weights}")
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
            )
            # Create CyclicLR callback
            cyclic_lr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=1000)
            history = model.fit(
                X_train_dev, y_train_dev,
                epochs=10,
                batch_size=32,
                class_weight=class_weights,
                validation_data=(X_val_dev, y_val_dev),
                callbacks=[early_stopping, cyclic_lr],
                verbose=0
            )
            if not history.history or 'loss' not in history.history or 'val_loss' not in history.history:
                logger.error(f"Client {device_name}: Training failed for round {self.round}, history invalid")
                return (model.get_weights(), len(X_train_dev), {"loss": 0.0, "val_loss": 0.0})
            weights = model.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], -1.0, 1.0)
            logger.info(f"Client {device_name}: Training completed for round {self.round}, loss: {history.history['loss'][-1]:.4f}, val_loss: {history.history['val_loss'][-1]:.4f}")
            logger.debug(f"Client {device_name}: fit returning weights with {len(weights)} arrays")
            result = (weights, len(X_train_dev), {"loss": float(history.history['loss'][-1]), "val_loss": float(history.history['val_loss'][-1])})
            return result

        def evaluate(self, parameters, config):
            logger.info(f"Client {device_name}: evaluate called with config: {config}")
            model.set_weights(parameters)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            if self.X_dev is None or self.y_dev is None or len(self.X_dev) == 0 or len(self.y_dev) == 0:
                logger.warning(f"Client {device_name}: No data for evaluation in round {self.round}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}
            if not np.issubdtype(self.y_dev.dtype, np.integer):
                logger.warning(f"Client {device_name}: Converting y_dev to integer type, was {self.y_dev.dtype}")
                self.y_dev = self.y_dev.astype(np.int32)
            if self.X_dev.shape[0] != len(self.y_dev):
                logger.error(f"Client {device_name}: Shape mismatch - X_dev shape {self.X_dev.shape} vs y_dev length {len(self.y_dev)}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}
            if np.min(self.y_dev) < 0 or np.max(self.y_dev) >= 6:
                logger.error(f"Client {device_name}: y_dev has labels out of range [0,5]: min={np.min(self.y_dev)}, max={np.max(self.y_dev)}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}
            logger.debug(f"Client {device_name}: Evaluating on X_dev shape {self.X_dev.shape}, y_dev shape {self.y_dev.shape}, y_dev dtype: {self.y_dev.dtype}")
            try:
                loss, acc = model.evaluate(self.X_dev, self.y_dev, verbose=0)
                y_pred = model.predict(self.X_dev, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                if len(y_pred_classes) != len(self.y_dev):
                    logger.error(f"Client {device_name}: Prediction length mismatch: {len(y_pred_classes)} vs {len(self.y_dev)}")
                    return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}
                f1 = f1_score(self.y_dev, y_pred_classes, average='macro')
            except Exception as e:
                logger.error(f"Client {device_name}: Error in evaluation: {e}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}
            logger.info(f"Client {device_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            self.metrics_history['loss'].append((self.round, float(loss)))
            self.metrics_history['accuracy'].append((self.round, float(acc)))
            self.metrics_history['f1_score'].append((self.round, float(f1)))
            metrics_file = f"C:/Users/rkjra/Desktop/FL/QFEDAVG/client_{device_name}_metrics.npy"
            np.save(metrics_file, self.metrics_history)
            logger.info(f"Client {device_name}: Saved evaluation metrics to {metrics_file}")
            return loss, len(self.X_dev), {"accuracy": acc, "f1_score": f1}

    server_address = "127.0.0.1:9000"
    logger.info(f"🔗 Attempting to connect to server at: {server_address}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"Client {device_name}: Connection attempt {attempt + 1}/{max_retries} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            channel = grpc.insecure_channel(server_address)
            grpc.channel_ready_future(channel).result(timeout=10)
            fl.client.start_numpy_client(
                server_address=server_address,
                client=QFedAvgClient(X_dev, y_dev)
            )
            channel.close()
            logger.info(f"🔌 gRPC channel closed for {device_name}")
            logger.info(f"✅ Client {device_name} completed.")
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error(f"❌ Client {device_name} failed after {max_retries} attempts: {e}")
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
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/X_test.npy", X_scaled_test)
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/y_test.npy", y_test)

    device_map = {'L': 0, 'M': 1, 'H': 2}

    device_code = device_map[device]
    X_dev = X_scaled_train[df_train['Device_Type'] == device_code]
    y_dev = y_train[df_train['Device_Type'] == device_code]
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Main: {device} class distribution: {class_dist}")
    min_samples = min(class_dist.values())

    if min_samples < 2 or device == 'L':
        max_samples = max(class_dist.values())
        X_dev_bal, y_dev_bal = X_dev.copy(), y_dev.copy()
        for class_label, count in class_dist.items():
            if count < max_samples:
                minority_X = X_dev[y_dev == class_label]
                minority_y = y_dev[y_dev == class_label]
                oversampled_X, oversampled_y = resample(
                    minority_X, minority_y, replace=True, n_samples=max_samples, random_state=42
                )
                X_dev_bal = np.vstack((X_dev_bal, oversampled_X))
                y_dev_bal = np.hstack((y_dev_bal, oversampled_y))
        logger.info(f"Main: {device} balanced with oversampling - {len(y_dev_bal)} samples")
    else:
        smote = SMOTE(k_neighbors=max(1, min_samples - 1), random_state=42)
        X_dev_bal, y_dev_bal = smote.fit_resample(X_dev, y_dev)
        logger.info(f"Main: {device} balanced with SMOTE - {len(y_dev_bal)} samples")

    logger.info(f"Starting client for device: {device}")
    run_client(device, X_dev_bal, y_dev_bal)
