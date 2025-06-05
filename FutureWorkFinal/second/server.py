import flwr as fl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import logging
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
import json

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Load server evaluation data
def load_server_data() -> Tuple[np.ndarray, np.ndarray, float]:
    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    X = df[["Air temperature [K]", "Process temperature [K]", 
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].values
    y = df["Target"].values
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Server: Input data contains NaN or inf values. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    input_variance = np.var(X)
    return X, y, input_variance

# Build the VAE model
def build_vae(input_dim: int = 5) -> Model:
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
    import tensorflow.keras.backend as K
    
    latent_dim = 2
    inputs = Input(shape=(input_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)
    z_log_var = Lambda(lambda x: K.clip(x, -2.0, 2.0))(z_log_var)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    decoder_h = Dense(32, activation='relu')
    decoder_h2 = Dense(64, activation='relu')
    decoder_out = Dense(input_dim, activation='sigmoid')

    h_decoded = decoder_h(z)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.2)(h_decoded)
    h_decoded = decoder_h2(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.1)(h_decoded)
    outputs = decoder_out(h_decoded)

    vae = Model(inputs, outputs, name='vae')
    beta = 0.001
    outputs_rescaled = (outputs * 2.0) - 1.0
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs_rescaled), axis=-1)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)
    vae.add_loss(total_loss)
    vae.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, clipnorm=0.5))
    return vae

# Custom strategy for clustered federated learning
class ClusteredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_clusters: int, fraction_fit: float, fraction_evaluate: float, min_fit_clients: int, min_evaluate_clients: int, min_available_clients: int, *args, **kwargs):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            *args,
            **kwargs
        )
        self.num_clusters = num_clusters
        self.cluster_weights = [None] * num_clusters
        self.cluster_counts = [0] * num_clusters
        self.X_eval, self.y_eval, self.input_variance = load_server_data()
        self.vae = build_vae(input_dim=self.X_eval.shape[1])
        # Initialize metrics storage
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "f1_score": [],
            "maintenance_accuracy": []
        }

    def num_clients_required(self, server_round: int) -> int:
        available_clients = self.min_available_clients
        fraction = self.fraction_fit
        num_clients = max(self.min_fit_clients, int(fraction * available_clients))
        return num_clients

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.SimpleClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        sample_size = self.num_clients_required(server_round)
        min_num_clients = self.min_fit_clients
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        client_clusters = [(client, idx % self.num_clusters) for idx, client in enumerate(clients)]
        logger.info(f"Round {server_round}: {len(clients)} clients connected")
        logger.info(f"Round {server_round}: Sampled {len(clients)} clients for fit: {[client.cid for client, _ in client_clusters]}")

        fit_configurations = []
        for client, cluster_id in client_clusters:
            has_all_weights = np.array([1.0 if all(w is not None for w in self.cluster_weights) else 0.0], dtype=np.float32)
            weights_to_send = [has_all_weights]
            
            cluster_params = fl.common.parameters_to_ndarrays(parameters) if server_round == 1 else self.cluster_weights[cluster_id]
            cluster_params = [np.clip(w, -10.0, 10.0) for w in cluster_params]
            weights_to_send.extend(cluster_params)

            if has_all_weights[0] > 0.5:
                for i in range(self.num_clusters):
                    if self.cluster_weights[i] is not None:
                        clipped_weights = [np.clip(w, -10.0, 10.0) for w in self.cluster_weights[i]]
                        weights_to_send.extend(clipped_weights)

            fit_ins = fl.common.FitIns(
                fl.common.ndarrays_to_parameters(weights_to_send),
                {"cluster_id": cluster_id}
            )
            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.SimpleClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        logger.info(f"Round {server_round}: Configuring evaluate...")
        clients = client_manager.all()
        logger.info(f"Round {server_round}: Waiting for {len(clients)} clients for evaluation, {client_manager.num_available()} available")
        
        if client_manager.num_available() < self.min_evaluate_clients:
            return []
        
        logger.info(f"Round {server_round}: {client_manager.num_available()} clients available for evaluation")
        clients = client_manager.sample(num_clients=self.min_evaluate_clients, min_num_clients=self.min_evaluate_clients)

        has_all_weights = np.array([1.0 if all(w is not None for w in self.cluster_weights) else 0.0], dtype=np.float32)
        eval_configurations = []
        for client in clients:
            cluster_id = int(client.cid.split(":")[-1]) % self.num_clusters
            weights_to_send = [has_all_weights]
            
            cluster_params = self.cluster_weights[cluster_id]
            if cluster_params is None:
                cluster_params = fl.common.parameters_to_ndarrays(parameters)
            cluster_params = [np.clip(w, -10.0, 10.0) for w in cluster_params]
            weights_to_send.extend(cluster_params)

            if has_all_weights[0] > 0.5:
                for i in range(self.num_clusters):
                    if self.cluster_weights[i] is not None:
                        clipped_weights = [np.clip(w, -10.0, 10.0) for w in self.cluster_weights[i]]
                        weights_to_send.extend(clipped_weights)

            eval_ins = fl.common.EvaluateIns(
                fl.common.ndarrays_to_parameters(weights_to_send),
                {"cluster_id": cluster_id}
            )
            eval_configurations.append((client, eval_ins))

        return eval_configurations

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        if not results:
            return None, {}

        self.cluster_counts = [0] * self.num_clusters
        cluster_updates = [[] for _ in range(self.num_clusters)]

        for client, fit_res in results:
            cluster_id = fit_res.metrics.get("cluster_id", 0)
            logger.info(f"Round {server_round}: Received fit results from client {client.cid} for cluster {cluster_id}")
            self.cluster_counts[cluster_id] += 1
            cluster_updates[cluster_id].append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        for cluster_id in range(self.num_clusters):
            if cluster_updates[cluster_id]:
                updates, weights = zip(*cluster_updates[cluster_id])
                aggregated_weights = [np.average([update[i] for update in updates], axis=0, weights=weights) for i in range(len(updates[0]))]
                aggregated_weights = [np.clip(w, -10.0, 10.0) for w in aggregated_weights]
                self.cluster_weights[cluster_id] = aggregated_weights
                logger.info(f"Round {server_round}: Aggregated weights for cluster {cluster_id}")
            else:
                logger.info(f"Round {server_round}: No updates for cluster {cluster_id}")

        logger.info(f"Server evaluate called for round {server_round}...")
        try:
            eval_weights = None
            for cluster_id in range(self.num_clusters):
                if self.cluster_weights[cluster_id] is not None:
                    eval_weights = self.cluster_weights[cluster_id]
                    break
            if eval_weights is None:
                raise ValueError("No cluster weights available for evaluation")

            eval_weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in eval_weights]
            self.vae.set_weights(eval_weights)

            reconstructions = self.vae.predict(self.X_eval, verbose=0)
            if np.any(np.isnan(reconstructions)) or np.any(np.isinf(reconstructions)):
                logger.warning("Server: VAE predictions contain NaN or inf during evaluation. Cleaning...")
                reconstructions = np.nan_to_num(reconstructions, nan=0.0, posinf=0.0, neginf=0.0)
            
            reconstructions_rescaled = (reconstructions * 2.0) - 1.0
            recon_errors = np.mean(np.square(self.X_eval - reconstructions_rescaled), axis=1)
            recon_errors = recon_errors / self.input_variance
            logger.info(f"Server: recon_errors stats - min: {np.min(recon_errors):.4f}, max: {np.max(recon_errors):.4f}, mean: {np.mean(recon_errors):.4f}")
            recon_errors = np.clip(recon_errors, 0, 100.0)
            clipped_percentage = np.mean(recon_errors == 100.0) * 100
            logger.info(f"Server: Percentage of recon_errors clipped to 100: {clipped_percentage:.2f}%")
            
            loss = float(np.mean(recon_errors))
            logger.info(f"Server evaluation - Loss: {loss:.4f}")
        except Exception as e:
            logger.error(f"Server evaluation failed: {str(e)}")
            raise

        return fl.common.ndarrays_to_parameters(eval_weights), {}

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}

        losses = []
        accuracies = []
        f1_scores = []
        maintenance_accuracies = []
        num_examples = 0

        for _, eval_res in results:
            num_examples += eval_res.num_examples
            losses.append(eval_res.loss * eval_res.num_examples)
            accuracies.append(eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples)
            f1_scores.append(eval_res.metrics.get("f1_score", 0.0) * eval_res.num_examples)
            maintenance_accuracies.append(eval_res.metrics.get("maintenance_accuracy", 0.0) * eval_res.num_examples)

        loss = sum(losses) / num_examples if num_examples > 0 else float('inf')
        accuracy = sum(accuracies) / num_examples if num_examples > 0 else 0.0
        f1_score_val = sum(f1_scores) / num_examples if num_examples > 0 else 0.0
        maintenance_accuracy = sum(maintenance_accuracies) / num_examples if num_examples > 0 else 0.0

        # Append metrics for this round
        self.metrics["loss"].append(float(loss))
        self.metrics["accuracy"].append(float(accuracy))
        self.metrics["f1_score"].append(float(f1_score_val))
        self.metrics["maintenance_accuracy"].append(float(maintenance_accuracy))

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score_val,
            "maintenance_accuracy": maintenance_accuracy
        }
        logger.info(f"Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1_score_val:.4f}, Maintenance Accuracy: {maintenance_accuracy:.4f}")
        return loss, metrics

    def save_metrics(self):
        with open("server_metrics.json", "w") as f:
            json.dump(self.metrics, f)
        logger.info("Server metrics saved to server_metrics.json")

# Start the server
def main():
    strategy = ClusteredFedAvg(
        num_clusters=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )

    # Save metrics after all rounds
    strategy.save_metrics()

if __name__ == "__main__":
    main()
