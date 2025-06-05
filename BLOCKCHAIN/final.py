import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
import logging
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = "C:/Users/rkjra/Desktop/FL/IFCA/"
METRICS_PATH = os.path.join(BASE_PATH, "metrics_history.npy")
CLIENT_METRICS = {
    'L': os.path.join(BASE_PATH, "client_L_metrics.npy"),
    'M': os.path.join(BASE_PATH, "client_M_metrics.npy"),
    'H': os.path.join(BASE_PATH, "client_H_metrics.npy")
}
X_TEST_PATH = os.path.join(BASE_PATH, "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_PATH, "y_test.npy")
CLUSTER_MODELS_PATH = os.path.join(BASE_PATH, "cluster_models.npy")

# Plot settings
plt.style.use('seaborn')
sns.set_palette("deep")

def build_vae(input_dim: int = 6) -> tuple[Model, Model]:
    """Build VAE and encoder models for latent space visualization."""
    logger.info("Building VAE model for visualization")
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
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    beta = 0.1
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, -1e5, 1e5)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.clip_by_value(kl_loss, -1e5, 1e5)
    total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

    vae.add_loss(total_loss)
    vae.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002, clipnorm=1.0))
    return vae, encoder

def load_metrics() -> Dict[str, List[tuple[int, float]]]:
    """Load server and client metrics from saved .npy files."""
    logger.info("Loading metrics...")
    metrics = {}
    
    # Server metrics
    try:
        server_metrics = np.load(METRICS_PATH, allow_pickle=True).item()
        metrics['server'] = server_metrics
        logger.info("Loaded server metrics")
    except Exception as e:
        logger.error(f"Failed to load server metrics: {str(e)}")
        metrics['server'] = {'loss': [], 'accuracy': [], 'f1_score': []}

    # Client metrics
    for device in ['L', 'M', 'H']:
        try:
            client_metrics = np.load(CLIENT_METRICS[device], allow_pickle=True).item()
            metrics[device] = client_metrics
            logger.info(f"Loaded metrics for client {device}")
        except Exception as e:
            logger.error(f"Failed to load metrics for client {device}: {str(e)}")
            metrics[device] = {'loss': [], 'accuracy': [], 'f1_score': []}

    return metrics

def plot_server_metrics(metrics: Dict[str, List[tuple[int, float]]], output_path: str):
    """Plot server loss, accuracy, and F1-score over rounds."""
    logger.info("Plotting server metrics")
    server_metrics = metrics['server']
    rounds = [r for r, _ in server_metrics['loss']]
    loss = [l for _, l in server_metrics['loss']]
    accuracy = [a for _, a in server_metrics['accuracy']]
    f1_score = [f for _, f in server_metrics['f1_score']]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    ax1.plot(rounds, loss, marker='o', label='Loss')
    ax1.set_title('Server Loss Over Rounds')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(rounds, accuracy, marker='o', label='Accuracy')
    ax2.set_title('Server Accuracy Over Rounds')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(rounds, f1_score, marker='o', label='F1-Score')
    ax3.set_title('Server F1-Score Over Rounds')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'server_metrics.png'))
    plt.close()
    logger.info("Saved server metrics plot")

def plot_client_metrics(metrics: Dict[str, List[tuple[int, float]]], output_path: str):
    """Plot client loss, accuracy, and F1-score over rounds."""
    logger.info("Plotting client metrics")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for device in ['L', 'M', 'H']:
        client_metrics = metrics[device]
        rounds = [r for r, _ in client_metrics['loss']]
        loss = [l for _, l in client_metrics['loss']]
        accuracy = [a for _, a in client_metrics['accuracy']]
        f1_score = [f for _, f in client_metrics['f1_score']]

        ax1.plot(rounds, loss, marker='o', label=f'Client {device}')
        ax2.plot(rounds, accuracy, marker='o', label=f'Client {device}')
        ax3.plot(rounds, f1_score, marker='o', label=f'Client {device}')

    ax1.set_title('Client Loss Over Rounds')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Client Accuracy Over Rounds')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Client F1-Score Over Rounds')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'client_metrics.png'))
    plt.close()
    logger.info("Saved client metrics plot")

def plot_cluster_assignments(output_path: str, client_clusters_history: Dict[int, Dict[str, int]] = None):
    """Plot cluster assignments as a heatmap (mock data if history not provided)."""
    logger.info("Plotting cluster assignments")
    # Mock cluster assignments if not provided (replace with actual data if available)
    if client_clusters_history is None:
        client_clusters_history = {
            1: {'L': 0, 'M': 1, 'H': 2},
            3: {'L': 0, 'M': 1, 'H': 2},
            6: {'L': 1, 'M': 1, 'H': 2},
            9: {'L': 1, 'M': 0, 'H': 2},
            10: {'L': 1, 'M': 0, 'H': 2}
        }
        logger.warning("Using mock cluster assignments. Replace with actual client_clusters_history for accurate visualization.")

    clients = ['L', 'M', 'H']
    rounds = sorted(client_clusters_history.keys())
    cluster_data = np.zeros((len(clients), len(rounds)))

    for j, round_num in enumerate(rounds):
        for i, client in enumerate(clients):
            cluster_data[i, j] = client_clusters_history[round_num].get(client, -1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cluster_data, cmap='viridis', annot=True, fmt='.0f',
                xticklabels=rounds, yticklabels=clients, cbar_kws={'label': 'Cluster ID'})
    ax.set_title('Client Cluster Assignments Over Rounds')
    ax.set_xlabel('Round')
    ax.set_ylabel('Client')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cluster_assignments.png'))
    plt.close()
    logger.info("Saved cluster assignments plot")

def plot_latent_space(output_path: str):
    """Plot 2D latent space for test data using the global VAE model."""
    logger.info("Plotting latent space")
    try:
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
        cluster_models = np.load(CLUSTER_MODELS_PATH, allow_pickle=True)

        # Compute global weights by averaging cluster models
        global_weights = []
        for layer in zip(*cluster_models):
            layer_weights = np.mean(np.array(layer), axis=0)
            global_weights.append(layer_weights)

        vae, encoder = build_vae(input_dim=X_test.shape[1])
        vae.set_weights(global_weights)

        z_mean, z_log_var, z = encoder.predict(X_test, verbose=0)
        labels = ['Normal' if y == 0 else 'Anomaly' for y in y_test]

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(z[:, 0], z[:, 1], c=y_test, cmap='coolwarm', alpha=0.6)
        ax.set_title('VAE Latent Space (Test Data)')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Anomaly'])
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_PATH, 'latent_space.png'))
        plt.close()
        logger.info("Saved latent space plot")
    except Exception as e:
        logger.error(f"Failed to plot latent space: {str(e)}")

def main():
    output_dir = os.path.join(BASE_PATH, "plots")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    metrics = load_metrics()
    plot_server_metrics(metrics, output_dir)
    plot_client_metrics(metrics, output_dir)
    plot_cluster_assignments(output_dir)  # Pass client_clusters_history if available
    plot_latent_space(output_dir)

if __name__ == "__main__":
    logger.info("Starting visualization script")
    main()
    logger.info("Visualization script completed")
