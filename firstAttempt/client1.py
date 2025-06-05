import flwr as fl
import time
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import grpc
from multiprocessing import Process
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import f1_score
import tensorflow as tf
import os

# Prevent threadpoolctl crash by limiting threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set up logging with DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enable Flower's DEBUG logging
fl.common.logger.configure(identifier="client")

def run_client(device_name, X_scaled_train_balanced, y_train_balanced, df_train_balanced, device_map):
    logger.info(f"🖥️ Running client for device: {device_name}")
    device_code = device_map[device_name]
    # Use balanced data with consistent indexing
    mask = df_train_balanced['Device_Type'] == device_code
    X_dev = X_scaled_train_balanced[mask]
    y_dev = y_train_balanced[mask]

    # Log class distribution and check for potential outliers
    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device_name}: Class distribution {class_dist}")
    if max(class_dist.values()) / min(class_dist.values()) > 10:
        logger.warning(f"Client {device_name}: High imbalance ratio detected: {max(class_dist.values()) / min(class_dist.values())}")

    def build_model():
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_dev.shape[1],), kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),  # Increased from 0.3 to 0.4
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(6, activation='softmax')
        ])
        return model

    model = build_model()

    # Compute class weights
    unique, counts = np.unique(y_dev, return_counts=True)
    class_weights = {k: (1.0 / v) * (len(y_dev) / len(unique)) for k, v in zip(unique, counts)}
    logger.info(f"Client {device_name}: Class weights: {class_weights}")

    class FedProxClient(fl.client.NumPyClient):
        def __init__(self):
            self.global_weights = None
            self.fit_called = False
            self.round = 0

        def get_parameters(self, config):
            logger.info(f"Client {device_name}: get_parameters called")
            return model.get_weights()

        def set_global_weights(self, weights):
            self.global_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights]

        def fit(self, parameters, config):
            logger.info(f"Client {device_name}: fit called with config: {config}")
            self.fit_called = True
            self.round += 1
            self.set_global_weights(parameters)
            model.set_weights(parameters)

            local_shapes = [w.shape for w in model.trainable_weights]
            global_shapes = [w.shape for w in self.global_weights]
            for i, (ls, gs) in enumerate(zip(local_shapes, global_shapes)):
                if ls != gs:
                    logger.warning(f"⚠️ Weight shape mismatch at index {i}: local={ls}, global={gs}")

            mu = config.get("mu", 0.01)  # Reduced back to 0.01 for more local flexibility
            initial_lr = 0.001
            lr = initial_lr * (0.9 ** (self.round - 1))  # Gradual exponential decay
            logger.info(f"Client {device_name}: Using LR {lr:.6f} for round {self.round}")

            def proximal_loss(y_true, y_pred):
                base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                prox_term = tf.constant(0.0, dtype=tf.float32)
                if self.global_weights is None:
                    logger.warning("⚠️ Skipping proximal term: global weights not set.")
                    return base_loss

                for i, (w_local, w_global) in enumerate(zip(model.trainable_weights, self.global_weights)):
                    if w_local.shape != w_global.shape:
                        logger.warning(f"⚠️ Skipping mismatch at index {i}: local shape={w_local.shape}, global shape={w_global.shape}")
                        continue
                    prox_term += tf.reduce_sum(tf.square(w_local - w_global))

                return base_loss + (mu / 2.0) * prox_term

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=0.5)  # Reduced clipnorm to 0.5
            model.compile(optimizer=optimizer, loss=proximal_loss, metrics=['accuracy'])
            history = model.fit(
                X_dev, y_dev,
                epochs=10,  # Reduced from 15 to 10
                batch_size=32,
                class_weight=class_weights,
                verbose=1
            )
            logger.info(f"Client {device_name}: training completed for round {self.round} with LR: {lr:.6f}")

            # Weight clipping
            weights = model.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], -0.5, 0.5)  # Reduced clipping range to [-0.5, 0.5]
            model.set_weights(weights)

            logger.info(f"Client {device_name}: returning weights with {len(weights)} arrays")
            return weights, len(X_dev), {"loss": float(history.history['loss'][-1])}

        def evaluate(self, parameters, config):
            logger.info(f"Client {device_name}: evaluate called")
            model.set_weights(parameters)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            loss, acc = model.evaluate(X_dev, y_dev, verbose=0)
            y_pred = model.predict(X_dev, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_dev, y_pred_classes, average='macro')
            logger.info(f"Client {device_name} evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            return loss, len(X_dev), {"accuracy": acc, "f1_score": f1}

    server_address = "127.0.0.1:9000"
    logger.info(f"🔗 Attempting to connect to server at: {server_address}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            channel = grpc.insecure_channel(server_address)
            grpc.channel_ready_future(channel).result(timeout=10)
            fl.client.start_numpy_client(
                server_address=server_address,
                client=FedProxClient()
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
    from multiprocessing import set_start_method
    try:
        set_start_method("spawn", force=True)
    except RuntimeError as e:
        logger.warning(f"Failed to set spawn method: {e}. Continuing with default.")

    try:
        df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
        df["Failure Type"] = df["Failure Type"].astype("category")
        df["Failure_Code"] = df["Failure Type"].cat.codes
        df["Device_Type"] = df["Type"].astype("category").cat.codes
    except FileNotFoundError:
        logger.error("❌ Dataset file not found. Please check the path.")
        exit(1)

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
    X = df[features]
    y = df['Failure_Code']

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    np.save("X_test.npy", X_scaled_test)
    np.save("y_test.npy", y_test)
    logger.info("Test set saved as 'X_test.npy' and 'y_test.npy'")

    device_map = {'L': 0, 'M': 1, 'H': 2}
    devices = ['L', 'M', 'H']
    X_scaled_train_balanced = []
    y_train_balanced = []
    df_train_balanced = []

    for device in devices:
        device_code = device_map[device]
        X_dev = X_scaled_train[df_train['Device_Type'] == device_code]
        y_dev = y_train[df_train['Device_Type'] == device_code]
        class_dist = dict(pd.Series(y_dev).value_counts())
        min_samples = min(class_dist.values())
        logger.info(f"Main: {device} class distribution: {class_dist}")
        if min_samples < 2:
            logger.warning(f"Main: {device} has class with < 2 samples. Using oversampling with replacement.")
            # Oversample minority classes to match the majority class
            max_samples = max(class_dist.values())
            X_dev_bal = X_dev.copy()
            y_dev_bal = y_dev.copy()
            for class_label, count in class_dist.items():
                if count < max_samples:
                    minority_X = X_dev[y_dev == class_label]
                    minority_y = y_dev[y_dev == class_label]
                    oversampled_X, oversampled_y = resample(minority_X, minority_y, replace=True, n_samples=max_samples, random_state=42)
                    X_dev_bal = np.vstack((X_dev_bal, oversampled_X))
                    y_dev_bal = np.hstack((y_dev_bal, oversampled_y))
            logger.info(f"Main: {device} balanced with oversampling - {len(y_dev_bal)} samples")
        else:
            k_neighbors = max(1, min_samples - 1)
            try:
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42, n_jobs=1)
                X_dev_bal, y_dev_bal = smote.fit_resample(X_dev, y_dev)
                logger.info(f"Main: {device} balanced with SMOTE - {len(y_dev_bal)} samples")
            except ValueError as e:
                logger.warning(f"Main: {device} SMOTE failed ({e}). Using original data.")
                X_dev_bal, y_dev_bal = X_dev, y_dev
        X_scaled_train_balanced.append(X_dev_bal)
        y_train_balanced.append(y_dev_bal)
        df_dev_bal = pd.DataFrame({'Device_Type': [device_code] * len(y_dev_bal)})
        df_train_balanced.append(df_dev_bal)

    X_scaled_train_balanced = np.vstack(X_scaled_train_balanced)
    y_train_balanced = np.hstack(y_train_balanced)
    df_train_balanced = pd.concat(df_train_balanced, ignore_index=True)

    processes = []
    for device in devices:
        p = Process(target=run_client, args=(device, X_scaled_train_balanced, y_train_balanced, df_train_balanced, device_map))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("All client executions completed.")
