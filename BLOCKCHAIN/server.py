import flwr as fl
import logging
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from typing import List, Dict, Tuple, Optional, Union
import grpc
import tensorflow.keras.backend as K
import socket
from datetime import datetime
from web3 import Web3
import os
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setStream(open(stream_handler.stream.fileno(), mode='w', encoding='utf-8', buffering=1))
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('server.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
fl.common.logger.configure(identifier="server")

SCRIPT_VERSION = "2025-05-13-v32"
logger.info(f"Running server.py version: {SCRIPT_VERSION}")

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

NUM_CLUSTERS = 3
cluster_models: List[List[np.ndarray]] = None
client_clusters: Dict[str, int] = {}
client_device_names: Dict[str, str] = {}  # New: Store cid -> device_name mappings
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}

class BlockchainServer:
    def __init__(self):
        logger.info("Initializing BlockchainServer...")
        self.web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
        try:
            if not self.web3.is_connected():
                logger.error(f"Failed to connect to Ethereum network at {WEB3_PROVIDER}")
                raise ConnectionError("Failed to connect to Ethereum network")
            chain_id = self.web3.eth.chain_id
            if chain_id != 17000:  # Holesky chain ID
                logger.error(f"Connected to wrong network. Expected Holesky (chain ID 17000), got chain ID {chain_id}")
                raise ConnectionError("Connected to wrong network")
            logger.info("Successfully connected to Holesky network")
            self.contract = self.web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
            self.account = self.web3.eth.account.from_key(PRIVATE_KEY)
            self.web3.eth.default_account = self.account.address
        except Exception as e:
            logger.error(f"BlockchainServer initialization failed: {str(e)}")
            logger.exception("Stack trace for blockchain initialization failure:")
            raise

    def check_client_registration(self, device_name: str) -> bool:
        try:
            is_available = self.contract.functions.checkAvailability(device_name).call()
            if is_available:
                logger.warning(f"Client {device_name} is not registered (name available)")
                return False
            owner = self.contract.functions.nameOwners(device_name).call()
            logger.info(f"Client {device_name} is registered with owner {owner}")
            return True
        except Exception as e:
            logger.error(f"Failed to check registration for client {device_name}: {str(e)}")
            logger.exception("Stack trace for check registration failure:")
            return False

def build_vae(input_dim: int = 6) -> Model:
    logger.info(f"Building VAE model with input_dim={input_dim}")
    latent_dim = 2
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

    decoder_h = Dense(48, activation='relu')
    decoder_h2 = Dense(96, activation='relu')
    decoder_out = Dense(input_dim, activation='linear')

    h_decoded = decoder_h(z)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.3)(h_decoded)
    h_decoded = decoder_h2(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.2)(h_decoded)
    outputs = decoder_out(h_decoded)

    vae = Model(inputs, outputs, name='vae')

    beta = 0.1
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, -1e5, 1e5)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.clip_by_value(kl_loss, -1e5, 1e5)
    total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

    vae.add_loss(total_loss)
    vae.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002, clipnorm=1.0))
    logger.info("VAE model compiled")
    return vae

def pretrain_model() -> List[np.ndarray]:
    logger.info("Pre-training base VAE model...")
    try:
        df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
        df["Failure_Code"] = df["Failure Type"].astype("category").cat.codes
        df["Device_Type"] = df["Type"].astype("category").cat.codes

        features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                    'Torque [Nm]', 'Tool wear [min]', 'Device_Type']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        vae = build_vae(input_dim=X_scaled.shape[1])
        vae.fit(X_scaled, X_scaled, epochs=20, batch_size=64, verbose=0)
        logger.info("Pre-training completed")
        return vae.get_weights()
    except Exception as e:
        logger.error(f"Failed to pre-train VAE: {str(e)}")
        logger.exception("Stack trace for pre-training failure:")
        raise

def check_port(host: str, port: int, retries: int = 3, delay: int = 2) -> bool:
    logger.info(f"Checking port {port} availability...")
    for attempt in range(retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.bind((host, port))
                logger.info(f"Port {port} is available")
                return True
            except socket.error as e:
                logger.warning(f"Attempt {attempt + 1}/{retries}: Port {port} is unavailable: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
    logger.error(f"Port {port} is not available after {retries} attempts")
    return False

class IFCAStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        initial_parameters: fl.common.typing.Parameters,
        blockchain_server: BlockchainServer,
        num_clusters: int = NUM_CLUSTERS,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3
    ):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.blockchain_server = blockchain_server
        self.num_clusters = num_clusters
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        logger.info(f"Initialized IFCAStrategy with {num_clusters} clusters")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[fl.common.typing.Parameters]:
        global cluster_models, client_clusters, client_device_names
        logger.info("Initializing parameters...")
        try:
            base_weights = fl.common.parameters_to_ndarrays(self.initial_parameters)
            cluster_models = [[w.copy() for w in base_weights] for _ in range(self.num_clusters)]
            client_clusters = {}
            client_device_names = {}
            logger.info(f"Initialized {self.num_clusters} cluster models")
            return self.initial_parameters
        except Exception as e:
            logger.error(f"Failed to initialize parameters: {str(e)}")
            logger.exception("Stack trace for initialization failure:")
            return None

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitIns]]:
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
            logger.error(f"Round {server_round}: Timeout, only {available_clients} clients available")
            return []

        try:
            sample_clients = client_manager.sample(
                num_clients=self.min_fit_clients,
                min_num_clients=self.min_fit_clients
            )
            logger.info(f"Round {server_round}: Sampled {len(sample_clients)} clients for fit: {[str(client.cid) for client in sample_clients]}")

            fit_ins = []
            for client in sample_clients:
                cid = str(client.cid)
                if cid in client_device_names:
                    device_name = client_device_names[cid]
                    if not self.blockchain_server.check_client_registration(device_name):
                        logger.error(f"Round {server_round}: Client {cid} ({device_name}) not registered on blockchain, skipping")
                        continue
                if cid not in client_clusters:
                    client_clusters[cid] = np.random.randint(0, self.num_clusters)
                cluster_id = client_clusters[cid]
                cluster_weights = cluster_models[cluster_id]
                quantized_weights = []
                for w in cluster_weights:
                    max_abs = np.max(np.abs(w))
                    if max_abs == 0:
                        quantized_weights.append(w.astype(np.int8))
                    else:
                        scaled = w * 127 / max_abs
                        quantized_weights.append(np.clip(np.round(scaled), -127, 127).astype(np.int8))
                parameters = fl.common.ndarrays_to_parameters(quantized_weights)
                config = {"round": server_round, "cluster_id": cluster_id}
                logger.info(f"Round {server_round}: Sending fit instructions to client {cid} (device_name: {client_device_names.get(cid, 'unknown')}) for cluster {cluster_id}")
                fit_ins.append((client, fl.common.FitIns(parameters, config)))
            if not fit_ins:
                logger.error(f"Round {server_round}: No registered clients available for fit")
            return fit_ins
        except Exception as e:
            logger.error(f"Round {server_round}: Failed to configure fit: {str(e)}")
            logger.exception("Stack trace for configure_fit failure:")
            return []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]]
    ) -> Tuple[Optional[fl.common.typing.Parameters], Dict[str, float]]:
        global cluster_models, client_device_names
        if not results:
            logger.error(f"Round {server_round}: No fit results received. Failures: {len(failures)}")
            return None, {}

        cluster_updates: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(self.num_clusters)]
        for client, res in results:
            cid = str(client.cid)
            if server_round == 1 and "device_name" in res.metrics:
                device_name = res.metrics["device_name"]
                client_device_names[cid] = device_name
                logger.info(f"Round {server_round}: Received device_name '{device_name}' for client {cid} in fit metrics")
                if not self.blockchain_server.check_client_registration(device_name):
                    logger.error(f"Round {server_round}: Client {cid} ({device_name}) not registered on blockchain, skipping")
                    continue
            cluster_id = client_clusters[cid]
            weights = fl.common.parameters_to_ndarrays(res.parameters)
            dequantized_weights = []
            for w in weights:
                max_abs = np.max(np.abs(w))
                if max_abs == 0:
                    dequantized_weights.append(w.astype(np.float32))
                else:
                    dequantized_weights.append((w.astype(np.float32) * max_abs / 127))
            cluster_updates[cluster_id].append((dequantized_weights, res.num_examples))
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

        try:
            np.save("C:/Users/rkjra/Desktop/FL/IFCA/cluster_models.npy", np.array(cluster_models, dtype=object), allow_pickle=True)
        except Exception as e:
            logger.error(f"Round {server_round}: Failed to save cluster models: {str(e)}")
            logger.exception("Stack trace for save failure:")

        return fl.common.ndarrays_to_parameters(cluster_models[0]), {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.EvaluateIns]]:
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
            logger.error(f"Round {server_round}: Timeout, only {available_clients} clients available for evaluation")
            return []

        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                sample_clients = client_manager.sample(
                    num_clients=self.min_evaluate_clients,
                    min_num_clients=self.min_evaluate_clients
                )
                logger.info(f"Round {server_round}: Attempt {attempt + 1}/{max_retries} - Sampled {len(sample_clients)} clients for evaluation: {[str(client.cid) for client in sample_clients]}")

                eval_ins = []
                send_all_weights = server_round % 3 == 0
                if send_all_weights:
                    all_cluster_weights = []
                    for weights in cluster_models:
                        quantized_weights = []
                        for w in weights:
                            max_abs = np.max(np.abs(w))
                            if max_abs == 0:
                                quantized_weights.append(w.astype(np.int8))
                            else:
                                scaled = w * 127 / max_abs
                                quantized_weights.append(np.clip(np.round(scaled), -127, 127).astype(np.int8))
                        all_cluster_weights.extend(quantized_weights)
                else:
                    all_cluster_weights = []

                for client in sample_clients:
                    cid = str(client.cid)
                    if cid in client_device_names:
                        device_name = client_device_names[cid]
                        if not self.blockchain_server.check_client_registration(device_name):
                            logger.error(f"Round {server_round}: Client {cid} ({device_name}) not registered on blockchain, skipping")
                            continue
                    cluster_id = client_clusters.get(cid, 0)
                    cluster_weights = cluster_models[cluster_id]
                    quantized_weights = []
                    for w in cluster_weights:
                        max_abs = np.max(np.abs(w))
                        if max_abs == 0:
                            quantized_weights.append(w.astype(np.int8))
                        else:
                            scaled = w * 127 / max_abs
                            quantized_weights.append(np.clip(np.round(scaled), -127, 127).astype(np.int8))
                    if send_all_weights:
                        parameters_to_send = fl.common.ndarrays_to_parameters(
                            [np.array([1.0])] + quantized_weights + all_cluster_weights
                        )
                    else:
                        parameters_to_send = fl.common.ndarrays_to_parameters(
                            [np.array([0.0])] + quantized_weights
                        )
                    config = {
                        "round": server_round,
                        "cluster_id": cluster_id
                    }
                    logger.info(f"Round {server_round}: Sending evaluate instructions to client {cid} (device_name: {client_device_names.get(cid, 'unknown')}) for cluster {cluster_id}, all_cluster_weights included: {send_all_weights}")
                    eval_ins.append((client, fl.common.EvaluateIns(parameters_to_send, config)))

                if eval_ins:
                    return eval_ins
                logger.error(f"Round {server_round}: Attempt {attempt + 1}/{max_retries} failed to configure evaluation (no registered clients)")
            except Exception as e:
                logger.error(f"Round {server_round}: Failed to configure evaluate (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.exception("Stack trace for configure_evaluate failure:")
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
        global metrics_history, client_clusters, client_device_names
        logger.info(f"Round {server_round}: Aggregate evaluate received {len(results)} results and {len(failures)} failures")

        for failure in failures:
            if isinstance(failure, tuple) and len(failure) == 2:
                client, res = failure
                try:
                    cid = str(client.cid)
                    logger.error(f"Round {server_round}: Evaluation failed for client {cid}. Failure details: {res}")
                except Exception as e:
                    logger.error(f"Round {server_round}: Could not retrieve failure details: {e}")
            else:
                logger.error(f"Round {server_round}: Evaluation failure (Exception): {str(failure)}")

        if not results:
            logger.error(f"Round {server_round}: No evaluation results received. Failures: {len(failures)}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        for client, res in results:
            cid = str(client.cid)
            if server_round == 1 and "device_name" in res.metrics:
                device_name = res.metrics["device_name"]
                client_device_names[cid] = device_name
                logger.info(f"Round {server_round}: Received device_name '{device_name}' for client {cid} in evaluate metrics")
                if not self.blockchain_server.check_client_registration(device_name):
                    logger.error(f"Round {server_round}: Client {cid} ({device_name}) not registered on blockchain, skipping")
                    continue

        send_all_weights = server_round % 3 == 0
        if send_all_weights:
            for client, res in results:
                cid = str(client.cid)
                if cid not in client_device_names:
                    logger.warning(f"Round {server_round}: Client {cid} has no device_name, skipping cluster reassignment")
                    continue
                client_losses = {}
                for cluster_id in range(self.num_clusters):
                    loss_key = f"cluster_loss_{cluster_id}"
                    loss = res.metrics.get(loss_key, float("inf"))
                    client_losses[cluster_id] = loss
                best_cluster = min(client_losses, key=client_losses.get)
                loss_threshold = 0.5
                if client_losses[best_cluster] < loss_threshold:
                    old_cluster = client_clusters.get(cid, 0)
                    client_clusters[cid] = best_cluster
                    logger.info(f"Round {server_round}: Client {cid} reassigned from cluster {old_cluster} to cluster {best_cluster} with losses: {client_losses}")
                else:
                    logger.info(f"Round {server_round}: Client {cid} not reassigned; best cluster loss {client_losses[best_cluster]:.4f} exceeds threshold {loss_threshold}")

        logger.info(f"Cluster assignments after round {server_round}: {client_clusters}")

        total_loss, total_accuracy, total_f1, total_maintenance_acc, total_samples = 0.0, 0.0, 0.0, 0.0, 0
        client_contributions = {}
        for client, res in results:
            cid = str(client.cid)
            client_loss = res.loss
            client_samples = res.num_examples
            total_loss += client_loss * client_samples
            total_accuracy += res.metrics["accuracy"] * client_samples
            total_f1 += res.metrics["f1_score"] * client_samples
            total_maintenance_acc += res.metrics.get("maintenance_accuracy", 0.0) * client_samples
            total_samples += client_samples
            client_contributions[cid] = {
                "loss": client_loss,
                "samples": client_samples,
                "accuracy": res.metrics["accuracy"],
                "f1_score": res.metrics["f1_score"]
            }
            logger.info(f"Round {server_round}: Client {cid} evaluation - Loss: {client_loss:.4f}, Accuracy: {res.metrics['accuracy']:.4f}, F1-Score: {res.metrics['f1_score']:.4f}")

        if total_samples == 0:
            logger.error(f"Round {server_round}: No valid evaluation results")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_f1 = total_f1 / total_samples
        avg_maintenance_acc = total_maintenance_acc / total_samples
        logger.info(f"Round {server_round} - Aggregated Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Score: {avg_f1:.4f}, Maintenance Accuracy: {avg_maintenance_acc:.4f}")

        logger.info(f"Round {server_round}: Client contributions to global metrics: {client_contributions}")

        metrics_history["loss"].append((server_round, avg_loss))
        metrics_history["accuracy"].append((server_round, avg_accuracy))
        metrics_history["f1_score"].append((server_round, avg_f1))
        try:
            np.save("C:/Users/rkjra/Desktop/FL/IFCA/metrics_history.npy", metrics_history, allow_pickle=True)
        except Exception as e:
            logger.error(f"Round {server_round}: Failed to save metrics history: {str(e)}")
            logger.exception("Stack trace for save failure:")

        return avg_loss, {"accuracy": avg_accuracy, "f1_score": avg_f1, "maintenance_accuracy": avg_maintenance_acc}

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.typing.Parameters
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        logger.info(f"Server evaluate called for round {server_round}...")
        try:
            X_test = np.load("C:/Users/rkjra/Desktop/FL/IFCA/X_test.npy")
            y_test = np.load("C:/Users/rkjra/Desktop/FL/IFCA/y_test.npy")
            logger.info(f"Test set class distribution: {dict(pd.Series(y_test).value_counts())}")

            global_weights = []
            for layer in zip(*cluster_models):
                layer_weights = np.mean(np.array(layer), axis=0)
                global_weights.append(layer_weights)

            vae = build_vae(input_dim=X_test.shape[1])
            vae.set_weights(global_weights)
            reconstructions = vae.predict(X_test, verbose=0)
            recon_errors = np.mean(np.square(X_test - reconstructions), axis=1)

            precisions, recalls, thresholds = precision_recall_curve(y_test, recon_errors)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            beta = 0.5
            f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
            best_threshold = thresholds[np.argmax(f_beta_scores)]
            y_pred = (recon_errors > best_threshold).astype(int)

            loss = np.mean(recon_errors)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            pred_dist = dict(pd.Series(y_pred).value_counts())
            logger.info(f"Server prediction distribution: {pred_dist}")
            logger.info(f"Server evaluation in round {server_round} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            if server_round == 10:
                logger.info(f"Final server evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            return loss, {"accuracy": acc, "f1_score": f1}
        except Exception as e:
            logger.error(f"Server evaluation failed in round {server_round}: {str(e)}")
            logger.exception("Stack trace for evaluation failure:")
            return None, {}

def run_server():
    logger.info(f"Starting server setup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    server_address = "127.0.0.1:9000"
    host, port = server_address.split(":")
    port = int(port)
    fallback_port = port + 1
    fallback_address = f"{host}:{fallback_port}"

    if not check_port(host, port):
        logger.warning(f"Trying fallback port {fallback_port}...")
        server_address = fallback_address
        port = fallback_port
        if not check_port(host, port):
            logger.error(f"Cannot start server. Both ports {port-1} and {port} are unavailable.")
            raise RuntimeError(f"Ports {port-1} and {port} are unavailable")

    try:
        logger.info("Initializing BlockchainServer...")
        blockchain_server = BlockchainServer()
        logger.info("Pre-training VAE model...")
        initial_weights = pretrain_model()
        logger.info("Converting initial weights to parameters...")
        initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)
        logger.info("Initializing IFCAStrategy...")
        strategy = IFCAStrategy(
            initial_parameters=initial_parameters,
            blockchain_server=blockchain_server,
            num_clusters=NUM_CLUSTERS,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3
        )
        logger.info(f"Starting Flower server at {server_address}...")
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024
        )
        logger.info("Flower server completed all rounds")
    except grpc.RpcError as e:
        logger.error(f"gRPC error starting server: {str(e)}")
        logger.exception("Stack trace for gRPC failure:")
        raise
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        logger.exception("Stack trace for server failure:")
        raise

if __name__ == "__main__":
    logger.info("Server script started")
    run_server()
    logger.info("Server script terminated")
