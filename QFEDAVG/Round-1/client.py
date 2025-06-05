import flwr as fl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import argparse
import grpc
import time
from datetime import datetime

# Configure logging (for terminal output only, since we'll use .npy for metrics)
logger = logging.getLogger("client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Version check to ensure correct script
SCRIPT_VERSION = "2025-04-20-v11"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

def build_model(input_dim):
    """Build a deep neural network model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
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
        def __init__(self):
            self.round = 0
            # Initialize metrics dictionary to store per-round evaluation metrics
            self.metrics_history = {
                'loss': [],
                'accuracy': [],
                'f1_score': []
            }

        def get_parameters(self, config):
            """Return model parameters."""
            logger.info(f"Client {device_name}: get_parameters called with config: {config}")
            weights = model.get_weights()
            logger.debug(f"Client {device_name}: get_parameters returning weights with {len(weights)} arrays")
            return weights

        def fit(self, parameters, config):
            """Train the model on local data."""
            self.round += 1
            logger.info(f"Client {device_name}: fit called for round {self.round} with config: {config}")
            model.set_weights(parameters)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.01 * (0.98 ** (self.round - 1)), clipnorm=1.0
            )
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            X_train_dev, X_val_dev, y_train_dev, y_val_dev = train_test_split(
                X_dev, y_dev, test_size=0.1, random_state=42
            )
            # Dynamically compute class weights for training data
            unique, counts = np.unique(y_train_dev, return_counts=True)
            class_weights = {k: (1.0 / v) * (len(y_train_dev) / len(unique)) for k, v in zip(unique, counts)}
            logger.info(f"Client {device_name}: Round {self.round} class weights: {class_weights}")
            # Callbacks for early stopping and learning rate reduction
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True, verbose=0
            )
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0
            )
            history = model.fit(
                X_train_dev, y_train_dev,
                epochs=20,
                batch_size=64,
                class_weight=class_weights,
                validation_data=(X_val_dev, y_val_dev),
                callbacks=[early_stopping, lr_scheduler],
                verbose=0
            )
            weights = model.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], -1.0, 1.0)
            logger.info(f"Client {device_name}: Training completed for round {self.round}, loss: {history.history['loss'][-1]:.4f}, val_loss: {history.history['val_loss'][-1]:.4f}")
            logger.debug(f"Client {device_name}: fit returning weights with {len(weights)} arrays")
            result = (weights, len(X_train_dev), {"loss": float(history.history['loss'][-1]), "val_loss": float(history.history['val_loss'][-1])})
            logger.debug(f"Client {device_name}: fit returning tuple: weights ({len(weights)} arrays), num_examples ({len(X_train_dev)}), metrics ({result[2]})")
            return result

        def evaluate(self, parameters, config):
            """Evaluate the model on local data and save metrics to .npy file."""
            logger.info(f"Client {device_name}: evaluate called with config: {config}")
            model.set_weights(parameters)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            loss, acc = model.evaluate(X_dev, y_dev, verbose=0)
            y_pred = model.predict(X_dev, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_dev, y_pred_classes, average='macro')
            logger.info(f"Client {device_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

            # Append metrics for this round
            self.metrics_history['loss'].append((self.round, float(loss)))
            self.metrics_history['accuracy'].append((self.round, float(acc)))
            self.metrics_history['f1_score'].append((self.round, float(f1)))

            # Save metrics to .npy file
            metrics_file = f"C:/Users/rkjra/Desktop/FL/QFEDAVG/client_{device_name}_metrics.npy"
            np.save(metrics_file, self.metrics_history)
            logger.info(f"Client {device_name}: Saved evaluation metrics to {metrics_file}")

            return loss, len(X_dev), {"accuracy": acc, "f1_score": f1}

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
                client=QFedAvgClient()
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
    # Parse command-line argument for device type
    parser = argparse.ArgumentParser(description="Run a federated learning client for a specific device.")
    parser.add_argument('--device', type=str, choices=['L', 'M', 'H'], required=True, 
                        help="Device type to run the client for (L, M, or H)")
    args = parser.parse_args()
    device = args.device

    # Load and preprocess dataset
    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    df["Failure_Code"] = df["Failure Type"].astype("category").cat.codes
    df["Device_Type"] = df["Type"].astype("category").cat.codes

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
    X, y = df[features], df['Failure_Code']
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/X_test.npy", X_scaled_test)
    np.save("C:/Users/rkjra/Desktop/FL/QFEDAVG/y_test.npy", y_test)

    # Device mapping
    device_map = {'L': 0, 'M': 1, 'H': 2}

    # Process data for the specified device
    device_code = device_map[device]
    X_dev = X_scaled_train[df_train['Device_Type'] == device_code]
    y_dev = y_train[df_train['Device_Type'] == device_code]
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Main: {device} class distribution: {class_dist}")
    min_samples = min(class_dist.values())
    
    # Balance data
    if min_samples < 2 or device == 'L':  # Oversampling for L and small classes
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
    else:  # SMOTE for M and H
        smote = SMOTE(k_neighbors=max(1, min_samples - 1), random_state=42)
        X_dev_bal, y_dev_bal = smote.fit_resample(X_dev, y_dev)
        logger.info(f"Main: {device} balanced with SMOTE - {len(y_dev_bal)} samples")

    # Run client for the specified device
    logger.info(f"Starting client for device: {device}")
    run_client(device, X_dev_bal, y_dev_bal)
