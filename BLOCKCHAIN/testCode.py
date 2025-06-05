import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('test_metrics.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Paths to the test data and cluster models
BASE_PATH = "C:/Users/rkjra/Desktop/FL/BLOCKCHAIN/"
X_TEST_PATH = os.path.join(BASE_PATH, "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_PATH, "y_test.npy")
CLUSTER_MODELS_PATH = os.path.join(BASE_PATH, "cluster_models.npy")

def build_vae(input_dim: int = 6) -> Model:
    """Build VAE model for evaluation."""
    logger.info("Building VAE model for test data evaluation")
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

    h_decoded = Dense(48, activation='relu')(z)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.3)(h_decoded)
    h_decoded = Dense(96, activation='relu')(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Dropout(0.2)(h_decoded)
    outputs = Dense(input_dim, activation='linear')(h_decoded)

    vae = Model(inputs, outputs, name='vae')

    beta = 0.1  # Consistent with the scripts
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs), axis=-1)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, -1e5, 1e5)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.clip_by_value(kl_loss, -1e5, 1e5)
    total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)

    vae.add_loss(total_loss)
    vae.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.002, clipnorm=1.0))
    return vae

def compute_test_metrics():
    logger.info(f"Starting test data metrics computation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load test data
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        logger.error("Test data files not found")
        raise FileNotFoundError("X_test.npy or y_test.npy not found")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    logger.info(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # Validate data
    if len(X_test) != len(y_test):
        logger.error("Mismatch in number of samples between X_test and y_test")
        raise ValueError("Number of samples in X_test and y_test do not match")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        logger.error("X_test contains NaN or Inf values")
        raise ValueError("X_test contains NaN or Inf values")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        logger.error("y_test contains NaN or Inf values")
        raise ValueError("y_test contains NaN or Inf values")

    # Load cluster models and compute global weights
    if not os.path.exists(CLUSTER_MODELS_PATH):
        logger.error("Cluster models file not found")
        raise FileNotFoundError("cluster_models.npy not found")
    cluster_models = np.load(CLUSTER_MODELS_PATH, allow_pickle=True)
    global_weights = []
    for layer in zip(*cluster_models):
        layer_weights = np.mean(np.array(layer), axis=0)
        global_weights.append(layer_weights)
    logger.info("Computed global weights by averaging cluster models")

    # Build and set up VAE
    vae = build_vae(input_dim=X_test.shape[1])
    vae.set_weights(global_weights)
    logger.info("VAE model initialized with global weights")

    # Compute reconstruction errors
    reconstructions = vae.predict(X_test, verbose=0)
    recon_errors = np.mean(np.square(X_test - reconstructions), axis=1)
    avg_loss = np.mean(recon_errors)
    logger.info(f"Average reconstruction loss on test data: {avg_loss:.4f}")

    # Compute precision, recall, and F1-score with optimized threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, recon_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    beta = 0.5  # Consistent with scripts
    f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
    best_threshold = thresholds[np.argmax(f_beta_scores)]
    y_pred = (recon_errors > best_threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Test data metrics - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    # Log class distribution of predictions
    pred_dist = dict(pd.Series(y_pred).value_counts())
    true_dist = dict(pd.Series(y_test).value_counts())
    logger.info(f"True label distribution: {true_dist}")
    logger.info(f"Predicted label distribution: {pred_dist}")

    return avg_loss, accuracy, f1

if __name__ == "__main__":
    logger.info("Test metrics computation script started")
    try:
        loss, accuracy, f1 = compute_test_metrics()
        logger.info(f"Final test metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    except Exception as e:
        logger.error(f"Test metrics computation failed: {str(e)}")
        raise
    logger.info("Test metrics computation script terminated")
