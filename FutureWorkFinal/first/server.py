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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fl.common.logger.configure(identifier="server")

SCRIPT_VERSION = "2025-05-10-v22"  # Updated version
logger.info(f"Running server.py version: {SCRIPT_VERSION}")

NUM_CLUSTERS = 3
cluster_models: List[List[np.ndarray]] = None
client_clusters: Dict[str, int] = {}
metrics_history = {"loss": [], "accuracy": [], "f1_score": []}

def build_vae(input_dim: int = 6) -> Model:
    """Build a Variational Autoencoder with KL weighting and larger architecture."""
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
    return vae

def pretrain_model() -> List[np.ndarray]:
    logger.info("Pre-training base VAE model...")
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
    logger.info("Pre-training completed.")
    return vae.get_weights()

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
            logger.info(f"Round {server_round}: Sending fit instructions to client {cid} for cluster {cluster_id}")
            fit_ins.append((client, fl.common.FitIns(parameters, config)))
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.typing.FitRes]]
    ) -> Tuple[Optional[fl.common.typing.Parameters], Dict[str, float]]:
        global cluster_models
        if not results:
            logger.warning(f"Round {server_round}: No fit results received. Failures: {len(failures)}")
            return None, {}

        cluster_updates: List[List[Tuple[List[np.ndarray], int]]] = [[] for _ in range(self.num_clusters)]
        for client, res in results:
            cid = str(client.cid)
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

        np.save("C:/Users/rkjra/Desktop/FL/IFCA/cluster_models.npy", np.array(cluster_models, dtype=object), allow_pickle=True)
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
                    all_cluster_weights.append(fl.common.ndarrays_to_parameters(quantized_weights))
            else:
                all_cluster_weights = []

            for client in sample_clients:
                cid = str(client.cid)
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
                        [np.array([1.0])] + quantized_weights + [w.tensors for w in all_cluster_weights]
                    )
                else:
                    parameters_to_send = fl.common.ndarrays_to_parameters(
                        [np.array([0.0])] + quantized_weights
                    )
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
        global metrics_history, client_clusters
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
            logger.warning(f"Round {server_round}: No evaluation results received. Failures: {len(failures)}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0}

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
            logger.warning(f"Round {server_round}: No valid evaluation results")
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
        np.save("C:/Users/rkjra/Desktop/FL/IFCA/metrics_history.npy", metrics_history, allow_pickle=True)
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
            return loss, {"accuracy": acc, "f1_score": f1}
        except Exception as e:
            logger.error(f"Server evaluation failed in round {server_round}: {e}")
            return None, {}

def run_server():
    logger.info("Starting server setup...")
    try:
        initial_weights = pretrain_model()
        logger.info("Converting initial weights to parameters...")
        initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)
        logger.info("Initializing IFCAStrategy...")
        strategy = IFCAStrategy(
            initial_parameters=initial_parameters,
            num_clusters=NUM_CLUSTERS,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3
        )
        logger.info("Starting Flower server...")
        fl.server.start_server(
            server_address="127.0.0.1:9000",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024
        )
        logger.info("Flower server completed all rounds.")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise

if __name__ == "__main__":
    run_server()
