import flwr as fl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import argparse
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE
import tensorflow.keras.backend as K
import shap
from web3 import Web3
import os
from dotenv import load_dotenv
import socket

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setStream(open(stream_handler.stream.fileno(), mode='w', encoding='utf-8', buffering=1))
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('client.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
fl.common.logger.configure(identifier="client")

SCRIPT_VERSION = "2025-05-13-v9"
logger.info(f"Running client.py version: {SCRIPT_VERSION}")

# Load environment variables
load_dotenv()

# Blockchain configuration
WEB3_PROVIDER = "https://rpc-holesky.rockx.com"
CONTRACT_ADDRESS = "0x78b47C6b26670A8723fd19CDcC7A1d9b1d300dd8"
PRIVATE_KEY = "2763525f5b8ec1e69a1eb9311026a5803b6b75b8d05439864b2018a5096d2bcb"
REGISTRATION_FEE = 5000000000000000

# NameRegistry contract ABI
CONTRACT_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "name", "type": "string"}],
        "name": "checkAvailability",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "name", "type": "string"}],
        "name": "claimName",
        "outputs": [],
        "payable": True,
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "name", "type": "string"}],
        "name": "nameOwners",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function"
    }
]

class BlockchainClient:
    def __init__(self, device_name: str):
        self.device_name = device_name
        logger.info(f"Client {device_name}: Initializing BlockchainClient...")
        self.web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
        try:
            if not self.web3.is_connected():
                logger.error(f"Client {device_name}: Failed to connect to Ethereum network at {WEB3_PROVIDER}")
                raise ConnectionError("Failed to connect to Ethereum network")
            chain_id = self.web3.eth.chain_id
            if chain_id != 17000:  # Holesky chain ID
                logger.error(f"Client {device_name}: Connected to wrong network. Expected Holesky (chain ID 17000), got chain ID {chain_id}")
                raise ConnectionError("Connected to wrong network")
            logger.info(f"Client {device_name}: Successfully connected to Holesky network")
            self.contract = self.web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
            self.account = self.web3.eth.account.from_key(PRIVATE_KEY)
            self.web3.eth.default_account = self.account.address
        except Exception as e:
            logger.error(f"Client {device_name}: BlockchainClient initialization failed: {str(e)}")
            logger.exception("Stack trace for blockchain initialization failure:")
            raise

    def check_registration(self) -> bool:
        try:
            is_available = self.contract.functions.checkAvailability(self.device_name).call()
            if is_available:
                logger.warning(f"Client {self.device_name}: Device is not registered")
                return False
            owner = self.contract.functions.nameOwners(self.device_name).call()
            logger.info(f"Client {self.device_name}: Device is registered with owner {owner}")
            return owner.lower() == self.account.address.lower()
        except Exception as e:
            logger.error(f"Client {self.device_name}: Failed to check registration: {str(e)}")
            logger.exception("Stack trace for check registration failure:")
            return False

    def register_device(self) -> bool:
        logger.info(f"Client {self.device_name}: Registering device on blockchain...")
        try:
            nonce = self.web3.eth.get_transaction_count(self.account.address)
            txn = self.contract.functions.claimName(self.device_name).build_transaction({
                'from': self.account.address,
                'value': int(REGISTRATION_FEE),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': nonce
            })
            logger.info(f"Client {self.device_name}: Building transaction with value={REGISTRATION_FEE} wei")
            signed_txn = self.web3.eth.account.sign_transaction(txn, PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                logger.info(f"Client {self.device_name}: Successfully registered device")
                return True
            else:
                logger.error(f"Client {self.device_name}: Registration transaction failed")
                return False
        except Exception as e:
            logger.error(f"Client {self.device_name}: Failed to register device: {str(e)}")
            logger.exception("Stack trace for registration failure:")
            return False

def balance_data_with_smote(X: np.ndarray, y: np.ndarray, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    class_dist = pd.Series(y).value_counts()
    logger.info(f"Client {device_name}: Class distribution before SMOTE: {dict(class_dist)}")
    
    min_samples = class_dist.min()
    if min_samples < 2:
        logger.warning(f"Client {device_name}: Some classes have fewer than 2 samples. Skipping SMOTE.")
        return X, y

    k_neighbors = min(5, min_samples - 1) if device_name == 'L' else min(3, min_samples - 1) if device_name == 'M' else min(7, min_samples - 1)
    logger.info(f"Client {device_name}: Using k_neighbors={k_neighbors} for SMOTE based on smallest class size ({min_samples} samples)")
    
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Client {device_name}: Class distribution after SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        return X_resampled, y_resampled
    except Exception as e:
        logger.error(f"Client {device_name}: SMOTE failed: {str(e)}. Falling back to original data.")
        return X, y

def build_vae(input_dim: int, device_name: str) -> tuple[Model, Model, Model]:
    logger.info(f"Client {device_name}: Building VAE model with input_dim={input_dim}")
    latent_dim = 2
    beta = 0.1 if device_name == 'L' else 0.05 if device_name == 'M' else 0.01

    inputs = Input(shape=(input_dim,))
    h = Dense(96, activation='relu')(inputs)
    h = BatchNormalization()(h)
    h = Dropout(0.4)(h)
    h = Dense(48, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,))
    h_decoded = Dense(48, activation='relu')(latent_inputs)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.3)(h_decoded)
    h_decoded = Dense(96, activation='relu')(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.2)(h_decoded)
    outputs = Dense(input_dim, activation='linear')(h_decoded)

    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(z)
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, -1e5, 1e5)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.clip_by_value(kl_loss, -1e5, 1e5)
    total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

    vae.add_loss(total_loss)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.002, clipnorm=1.0)
    vae.compile(optimizer=optimizer)
    vae.encoder = encoder
    vae.decoder = decoder

    logger.info(f"Client {device_name}: VAE model compiled with beta={beta}")
    return vae, encoder, decoder

def lr_schedule(epoch: int, device_name: str) -> float:
    initial_lr = 0.002
    drop_rate = 0.5
    epochs_drop = 10.0 if device_name == 'L' else 8.0 if device_name == 'M' else 12.0
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    logger.info(f"Client {device_name}: Learning rate for epoch {epoch}: {lr:.6f}")
    return lr

class QLearningAgent:
    def __init__(self, state_bins: int, actions: int):
        self.state_bins = state_bins
        self.actions = actions
        self.q_table = np.zeros((state_bins, actions))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.state_boundaries = None

    def set_state_boundaries(self, errors: np.ndarray):
        error_min, error_max = np.min(errors), np.max(errors)
        if error_min == error_max:
            error_max += 1e-10
        self.state_boundaries = np.linspace(error_min, error_max, self.state_bins)

    def discretize_state(self, error: float) -> int:
        if self.state_boundaries is None:
            return 0
        return np.digitize(error, self.state_boundaries, right=True).clip(0, self.state_bins - 1)

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

def check_server_availability(host: str, port: int, timeout: float = 2.0) -> bool:
    logger.info(f"Checking server availability at {host}:{port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            logger.info(f"Server at {host}:{port} is reachable")
            return True
        except socket.error as e:
            logger.warning(f"Server at {host}:{port} is not reachable: {e}")
            return False

def run_client(device_name: str, X_dev: np.ndarray, y_dev: np.ndarray, failure_labels: np.ndarray) -> None:
    logger.info(f"Client {device_name}: Running client at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    bc_client = BlockchainClient(device_name)
    if not bc_client.check_registration():
        logger.info(f"Client {device_name}: Device not registered. Attempting to register...")
        if not bc_client.register_device():
            logger.error(f"Client {device_name}: Failed to register device ID. Aborting.")
            return
    logger.info(f"Client {device_name}: Device ID verified on blockchain")

    if len(X_dev) == 0 or len(y_dev) == 0 or len(failure_labels) == 0:
        logger.error(f"Client {device_name}: No data available after filtering. Skipping client.")
        return

    logger.info(f"Client {device_name}: Input feature range - min: {np.min(X_dev):.4f}, max: {np.max(X_dev):.4f}")
    if np.any(np.isnan(X_dev)) or np.any(np.isinf(X_dev)):
        logger.error(f"Client {device_name}: Input data X_dev contains NaN or Inf values.")
        raise ValueError("Input data X_dev contains NaN or Inf values.")
    if np.any(np.isnan(y_dev)) or np.any(np.isinf(y_dev)):
        logger.error(f"Client {device_name}: Labels y_dev contain NaN or Inf values.")
        raise ValueError("Labels y_dev contain NaN or Inf values.")

    vae, encoder, decoder = build_vae(X_dev.shape[1], device_name)
    X_labeled, y_labeled, failure_labels_labeled = X_dev, y_dev, failure_labels

    logger.info(f"Client {device_name}: Labeled data: {len(X_labeled)} samples")

    q_agent = QLearningAgent(state_bins=10, actions=2)

    class IFCAClient(fl.client.NumPyClient):
        def __init__(self, X_labeled: np.ndarray, y_labeled: np.ndarray, failure_labels: np.ndarray):
            self.round = 0
            self.X_labeled = X_labeled
            self.y_labeled = y_labeled
            self.failure_labels = failure_labels
            self.metrics_history = {
                'loss': [],
                'accuracy': [],
                'f1_score': []
            }
            self.device_name = device_name
            logger.info(f"Client {device_name}: Initialized IFCAClient instance")

        def get_parameters(self, config: dict) -> list[np.ndarray]:
            logger.info(f"Client {self.device_name}: get_parameters called with config: {config}")
            weights = vae.get_weights()
            weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in weights]
            quantized_weights = []
            for w in weights:
                max_abs = np.max(np.abs(w))
                if max_abs == 0:
                    quantized_weights.append(w.astype(np.int8))
                else:
                    scaled = w * 127 / max_abs
                    quantized_weights.append(np.clip(np.round(scaled), -127, 127).astype(np.int8))
            logger.info(f"Client {self.device_name}: Sending {len(quantized_weights)} quantized weight arrays to server")
            return quantized_weights

        def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
            self.round += 1
            logger.info(f"Client {self.device_name}: fit called for round {self.round} with config: {config}")
            cluster_id = config.get("cluster_id", 0)
            logger.info(f"Client {self.device_name}: Assigned to cluster {cluster_id}")

            if not isinstance(parameters, list):
                logger.error(f"Client {self.device_name}: Expected list of weights in fit, got {type(parameters)}")
                raise ValueError(f"Expected list of weights in fit, got {type(parameters)}")
            
            dequantized_weights = []
            for w in parameters:
                max_abs = np.max(np.abs(w))
                if max_abs == 0:
                    dequantized_weights.append(w.astype(np.float32))
                else:
                    dequantized_weights.append((w.astype(np.float32) * max_abs / 127))
            
            try:
                vae.set_weights(dequantized_weights)
            except Exception as e:
                logger.error(f"Client {self.device_name}: Failed to set weights in fit: {str(e)}")
                raise

            num_samples = len(self.X_labeled)
            epochs = 5
            logger.info(f"Client {self.device_name}: Training for {epochs} epochs with {num_samples} samples")

            lr_scheduler = LearningRateScheduler(lambda epoch: lr_schedule(epoch, self.device_name))
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
            history = vae.fit(
                self.X_labeled, self.X_labeled,
                epochs=epochs,
                batch_size=64,
                callbacks=[lr_scheduler, early_stopping],
                verbose=0
            )

            final_loss = history.history['loss'][-1]
            logger.info(f"Client {self.device_name}: Training completed for round {self.round}, loss: {final_loss:.4f}")
            
            metrics = {"loss": float(final_loss)}
            if self.round == 1:
                metrics["device_name"] = self.device_name
                logger.info(f"Client {self.device_name}: Sending device_name '{self.device_name}' in fit metrics for round 1")
            
            return self.get_parameters(config), len(self.X_labeled), metrics

        def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
            logger.info(f"Client {self.device_name}: evaluate called for round {self.round} with config: {config}")
            cluster_id = config.get("cluster_id", 0)
            logger.info(f"Client {self.device_name}: Evaluating with cluster {cluster_id} model")

            if not isinstance(parameters, list):
                logger.error(f"Client {self.device_name}: Expected list of weights in evaluate, got {type(parameters)}")
                raise ValueError(f"Expected list of weights in evaluate, got {type(parameters)}")

            has_all_weights = bool(parameters[0][0] > 0.5)
            logger.info(f"Client {self.device_name}: Has all_cluster_weights: {has_all_weights}")

            expected_weights_len = len(vae.get_weights())
            cluster_weights = parameters[1:1 + expected_weights_len]
            if len(cluster_weights) != expected_weights_len:
                logger.error(f"Client {self.device_name}: Expected {expected_weights_len} weights, got {len(cluster_weights)}")
                raise ValueError(f"Expected {expected_weights_len} weights, got {len(cluster_weights)}")

            dequantized_weights = []
            for w in cluster_weights:
                max_abs = np.max(np.abs(w))
                if max_abs == 0:
                    dequantized_weights.append(w.astype(np.float32))
                else:
                    dequantized_weights.append((w.astype(np.float32) * max_abs / 127))

            try:
                vae.set_weights(dequantized_weights)
            except Exception as e:
                logger.error(f"Client {self.device_name}: Failed to set weights in evaluate: {str(e)}")
                raise

            if len(self.X_labeled) == 0:
                logger.warning(f"Client {self.device_name}: No data for evaluation in round {self.round}")
                return 0.0, 0, {"accuracy": 0.0, "f1_score": 0.0}

            logger.info(f"Client {self.device_name}: Evaluating on dataset with {len(self.X_labeled)} samples")

            start_time = time.time()
            reconstructions = vae.predict(self.X_labeled, verbose=0)
            recon_errors = np.mean(np.square(self.X_labeled - reconstructions), axis=1)
            avg_recon_error = np.mean(recon_errors)
            logger.info(f"Client {self.device_name}: Average reconstruction error: {avg_recon_error:.4f}")
            eval_time = time.time() - start_time
            logger.info(f"Client {self.device_name}: Primary evaluation took {eval_time:.2f} seconds")

            if len(self.X_labeled) >= 30:
                subset_size = min(30, len(self.X_labeled))
                X_subset = self.X_labeled[:subset_size]
                def recon_error_predict(X):
                    return np.mean(np.square(vae.predict(X, verbose=0) - X), axis=1)
                explainer = shap.KernelExplainer(recon_error_predict, X_subset)
                shap_values = explainer.shap_values(X_subset, nsamples=10, l1_reg="aic", eps=1e-5)
                feature_names = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                                 'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
                shap_summary = {name: float(np.mean(np.abs(shap_values[:, i]))) for i, name in enumerate(feature_names)}
                logger.info(f"Client {self.device_name}: SHAP feature importance for anomalies: {shap_summary}")

            z_mean, z_log_var, z = encoder.predict(self.X_labeled, verbose=0)
            normal_mask = self.failure_labels == 0
            anomaly_mask = self.failure_labels == 1
            if len(z_normal := z[normal_mask]) > 0:
                z_normal_mean = np.mean(z_normal, axis=0)
                z_normal_var = np.var(z_normal, axis=0)
                logger.info(f"Client {self.device_name}: Latent space (normal) - Mean: {z_normal_mean}, Variance: {z_normal_var}")
            if len(z_anomaly := z[anomaly_mask]) > 0:
                z_anomaly_mean = np.mean(z_anomaly, axis=0)
                z_anomaly_var = np.var(z_anomaly, axis=0)
                logger.info(f"Client {self.device_name}: Latent space (anomaly) - Mean: {z_anomaly_mean}, Variance: {z_anomaly_var}")

            precisions, recalls, thresholds = precision_recall_curve(self.failure_labels, recon_errors)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            beta = 0.5
            f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
            best_threshold = thresholds[np.argmax(f_beta_scores)]
            y_pred = (recon_errors > best_threshold).astype(int)

            acc = accuracy_score(self.failure_labels, y_pred)
            f1 = f1_score(self.failure_labels, y_pred, average='weighted')

            q_agent.set_state_boundaries(recon_errors)
            maintenance_decisions = []
            for i, error in enumerate(recon_errors):
                state = q_agent.discretize_state(error)
                action = q_agent.choose_action(state)
                maintenance_decisions.append(action)
                actual_failure = self.failure_labels[i]
                if action == 1 and actual_failure == 1:
                    reward = 1
                elif action == 1 and actual_failure == 0:
                    reward = -1
                elif action == 0 and actual_failure == 1:
                    reward = -2
                else:
                    reward = 0
                next_state = q_agent.discretize_state(error)
                q_agent.update(state, action, reward, next_state)

            maintenance_acc = accuracy_score(self.failure_labels, maintenance_decisions)
            logger.info(f"Client {self.device_name}: Maintenance scheduling accuracy: {maintenance_acc:.4f}")

            class_acc = {}
            class_f1 = {}
            for label in range(2):
                mask = self.failure_labels == label
                if mask.sum() > 0:
                    class_acc[label] = accuracy_score(self.failure_labels[mask], y_pred[mask])
                    class_f1[label] = f1_score(self.failure_labels[mask], y_pred[mask], average='weighted')
            logger.info(f"Client {self.device_name}: Class-wise accuracy: {class_acc}")
            logger.info(f"Client {self.device_name}: Class-wise F1-score: {class_f1}")

            cluster_losses = {}
            if has_all_weights:
                all_cluster_weights_flat = parameters[1 + expected_weights_len:]
                weights_per_cluster = expected_weights_len
                all_cluster_weights = [
                    all_cluster_weights_flat[i:i + weights_per_cluster]
                    for i in range(0, len(all_cluster_weights_flat), weights_per_cluster)
                ]
                logger.info(f"Client {self.device_name}: Received {len(all_cluster_weights)} cluster weights sets")

                for cluster_idx, cluster_weights in enumerate(all_cluster_weights):
                    try:
                        if len(cluster_weights) != expected_weights_len:
                            logger.error(f"Client {self.device_name}: Cluster {cluster_idx} has incorrect number of weights: {len(cluster_weights)}")
                            cluster_losses[f"cluster_loss_{cluster_idx}"] = float("inf")
                            continue

                        dequantized_cluster_weights = []
                        for w in cluster_weights:
                            max_abs = np.max(np.abs(w))
                            if max_abs == 0:
                                dequantized_cluster_weights.append(w.astype(np.float32))
                            else:
                                dequantized_cluster_weights.append((w.astype(np.float32) * max_abs / 127))

                        start_time = time.time()
                        vae.set_weights(dequantized_cluster_weights)
                        reconstructions = vae.predict(self.X_labeled, verbose=0)
                        cluster_loss = np.mean(np.square(self.X_labeled - reconstructions))
                        cluster_eval_time = time.time() - start_time
                        cluster_losses[f"cluster_loss_{cluster_idx}"] = float(cluster_loss)
                        logger.info(f"Client {self.device_name}: Loss for cluster {cluster_idx}: {cluster_loss:.4f}, took {cluster_eval_time:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Client {self.device_name}: Failed to evaluate cluster {cluster_idx}: {str(e)}")
                        cluster_losses[f"cluster_loss_{cluster_idx}"] = float("inf")

            loss = np.mean(recon_errors)
            logger.info(f"Client {self.device_name} - Loss: {loss:.4f}, Anomaly Detection Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            self.metrics_history['loss'].append((self.round, float(loss)))
            self.metrics_history['accuracy'].append((self.round, float(acc)))
            self.metrics_history['f1_score'].append((self.round, float(f1)))
            metrics_file = f"C:/Users/rkjra/Desktop/FL/IFCA/client_{self.device_name}_metrics.npy"
            np.save(metrics_file, self.metrics_history)

            metrics = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "maintenance_accuracy": float(maintenance_acc)
            }
            metrics.update(cluster_losses)
            if self.round == 1:
                metrics["device_name"] = self.device_name
                logger.info(f"Client {self.device_name}: Sending device_name '{self.device_name}' in evaluate metrics for round 1")
            
            logger.info(f"Client {self.device_name}: Returning evaluation results with metrics: {metrics}")
            return loss, len(self.X_labeled), metrics

    server_address = "127.0.0.1:9000"
    host, port = server_address.split(":")
    port = int(port)
    logger.info(f"Client {device_name}: Attempting to connect to server at: {server_address}")
    max_retries = 12
    retry_delay = 30
    for attempt in range(max_retries):
        logger.info(f"Client {device_name}: Connection attempt {attempt + 1}/{max_retries} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if check_server_availability(host, port):
            try:
                logger.info(f"Client {device_name}: Starting Flower client...")
                fl.client.start_numpy_client(
                    server_address=server_address,
                    client=IFCAClient(X_labeled, y_labeled, failure_labels_labeled),
                    grpc_max_message_length=1024*1024*1024
                )
                logger.info(f"Client {device_name}: Successfully completed all rounds")
                break
            except Exception as e:
                logger.warning(f"Client {device_name}: gRPC connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
        else:
            logger.warning(f"Client {device_name}: Server not available for attempt {attempt + 1}/{max_retries}")
        if attempt < max_retries - 1:
            logger.info(f"Client {device_name}: Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error(f"Client {device_name}: Failed after {max_retries} attempts")
            raise RuntimeError(f"Failed to connect to server at {server_address} after {max_retries} attempts")

if __name__ == "__main__":
    logger.info("Client script started")
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--device', type=str, choices=['L', 'M', 'H'], required=True)
    args = parser.parse_args()
    device = args.device

    try:
        df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    except Exception as e:
        logger.error(f"Failed to load predictive_maintenance.csv: {str(e)}")
        raise

    df["Failure_Code"] = df["Failure Type"].astype("category").cat.codes
    df["Device_Type"] = df["Type"].astype("category").cat.codes
    df["Failure_Label"] = (df["Failure Type"] != "No Failure").astype(int)

    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
    X, y = df[features], df['Failure_Code']
    failure_labels = df['Failure_Label'].values

    df_resampled = pd.DataFrame(X, columns=features)
    df_resampled["Failure_Code"] = y
    df_resampled["Device_Type"] = df_resampled["Device_Type"].astype(int)
    df_resampled["Failure_Label"] = failure_labels

    X_train, X_test, y_train, y_test, failure_labels_train, failure_labels_test, df_train, df_test = train_test_split(
        X, y, failure_labels, df_resampled, test_size=0.2, random_state=42, stratify=failure_labels
    )

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    try:
        np.save("C:/Users/rkjra/Desktop/FL/IFCA/X_test.npy", X_scaled_test)
        np.save("C:/Users/rkjra/Desktop/FL/IFCA/y_test.npy", failure_labels_test)
    except Exception as e:
        logger.error(f"Failed to save test data: {str(e)}")
        raise

    device_map = {'L': 0, 'M': 1, 'H': 2}
    device_code = device_map[device]
    logger.info(f"Client {device}: Device mapped to Device_Type code: {device_code}")
    logger.info(f"Client {device}: Device_Type distribution in training set: {dict(df_train['Device_Type'].value_counts())}")

    device_mask = df_train['Device_Type'] == device_code
    X_dev_unscaled = X_train[device_mask]
    y_dev = y_train[device_mask]
    failure_labels_dev = failure_labels_train[device_mask]

    X_dev_unscaled, y_dev = balance_data_with_smote(X_dev_unscaled, y_dev, device)
    failure_labels_dev = np.where(y_dev == 1, 0, 1)
    logger.info(f"Client {device}: Failure labels distribution after SMOTE: {dict(pd.Series(failure_labels_dev).value_counts())}")

    df_dev = pd.DataFrame(X_dev_unscaled, columns=features)
    df_dev["Failure_Code"] = y_dev
    df_dev["Device_Type"] = df_dev["Device_Type"].astype(int)
    df_dev["Failure_Label"] = failure_labels_dev

    X_dev = scaler.transform(X_dev_unscaled)

    class_dist = dict(pd.Series(y_dev).value_counts())
    logger.info(f"Client {device}: Class distribution after SMOTE: {class_dist}")

    logger.info(f"Client {device}: Starting client with {len(y_dev)} samples")
    run_client(device, X_dev, y_dev, failure_labels_dev)
    logger.info(f"Client {device}: Client script terminated")
