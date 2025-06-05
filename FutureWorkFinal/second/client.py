import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import json
import logging
from imblearn.over_sampling import SMOTE
import shap
import tensorflow.keras.backend as K

# Configure logging
logger = logging.getLogger("client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Load and preprocess the dataset
def load_data(client_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    df_client = df[df["Type"] == client_type]
    X = df_client[["Air temperature [K]", "Process temperature [K]", 
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].values
    y = df_client["Target"].values
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning(f"Client {client_type}: Input data contains NaN or inf values. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logger.info(f"Client {client_type}: Class distribution after SMOTE: {dict(pd.Series(y_train).value_counts())}")
    input_variance = np.var(X_train)
    return X_train, X_test, y_train, y_test, input_variance

# Build VAE model
def build_vae(input_dim: int = 5) -> tuple[Model, Model, Model]:
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
    return vae, Model(inputs, [z_mean, z_log_var, z], name='encoder'), Model(decoder_input := Input(shape=(latent_dim,)), decoder_out(BatchNormalization()(Dropout(0.1)(BatchNormalization()(Dropout(0.2)(decoder_h2(BatchNormalization()(Dropout(0.2)(decoder_h(decoder_input))))))))), name='decoder')

# Learning rate scheduler
def lr_schedule(epoch: int) -> float:
    initial_lr = 0.0005
    decay = 0.01
    lr = initial_lr * (1.0 / (1.0 + decay * epoch))
    return lr

# Q-Learning Agent for maintenance scheduling
class QLearningAgent:
    def __init__(self, state_bins: int = 10, actions: int = 2, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.state_bins = state_bins
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_bins, actions))

    def discretize_state(self, reconstruction_error: float) -> int:
        bins = np.linspace(0, 10, self.state_bins)
        return np.digitize(reconstruction_error, bins) - 1

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )

# Flower client class
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_type: str):
        self.client_type = client_type
        self.X_train, self.X_test, self.y_train, self.y_test, self.input_variance = load_data(client_type)
        self.vae, self.encoder, self.decoder = build_vae(input_dim=self.X_train.shape[1])
        self.q_agent = QLearningAgent()
        self.metrics = {"train_loss": [], "train_accuracy": [], "train_f1_score": [],
                       "test_loss": [], "test_accuracy": [], "test_f1_score": [],
                       "maintenance_accuracy": []}

    def get_parameters(self, config):
        weights = self.vae.get_weights()
        weights = [np.clip(w, -10.0, 10.0) for w in weights]
        weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in weights]
        quantized_weights = []
        for w in weights:
            max_abs = np.max(np.abs(w))
            if max_abs == 0:
                quantized_weights.append(w.astype(np.int8))
            else:
                scaled = w * 127 / max_abs
                quantized_weights.append(np.clip(np.round(scaled), -127, 127).astype(np.int8))
        return quantized_weights

    def fit(self, parameters, config):
        logger.info(f"Client {self.client_type}: Received {len(parameters)} parameters for fit")
        expected_weights_len = len(self.vae.get_weights())
        has_all_weights = parameters[0][0] > 0.5
        logger.info(f"Client {self.client_type}: has_all_weights: {has_all_weights}, expected weights length: {expected_weights_len}")

        cluster_weights = parameters[1:expected_weights_len + 1]
        if len(cluster_weights) != expected_weights_len:
            logger.error(f"Client {self.client_type}: Expected {expected_weights_len} weights, but got {len(cluster_weights)}")
            raise ValueError(f"Expected {expected_weights_len} weights, but got {len(cluster_weights)}")

        dequantized_weights = []
        for w in cluster_weights:
            max_abs = np.max(np.abs(w))
            if max_abs == 0:
                dequantized_weights.append(w.astype(np.float32))
            else:
                dequantized_weights.append((w.astype(np.float32) * max_abs / 127))
        dequantized_weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in dequantized_weights]
        dequantized_weights = [np.clip(w, -10.0, 10.0) for w in dequantized_weights]
        logger.info(f"Client {self.client_type}: Dequantized weights length: {len(dequantized_weights)}")
        self.vae.set_weights(dequantized_weights)

        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
        history = self.vae.fit(
            self.X_train, self.X_train,
            epochs=5,
            batch_size=32,
            callbacks=[lr_scheduler, early_stopping],
            verbose=0
        )
        loss = float(history.history["loss"][-1])
        self.metrics["train_loss"].append(loss)

        reconstructions = self.vae.predict(self.X_train, verbose=0)
        if np.any(np.isnan(reconstructions)) or np.any(np.isinf(reconstructions)):
            logger.warning(f"Client {self.client_type}: VAE predictions contain NaN or inf. Cleaning...")
            reconstructions = np.nan_to_num(reconstructions, nan=0.0, posinf=0.0, neginf=0.0)
        
        reconstructions_rescaled = (reconstructions * 2.0) - 1.0
        recon_errors = np.mean(np.square(self.X_train - reconstructions_rescaled), axis=1)
        if np.any(np.isnan(recon_errors)) or np.any(np.isinf(recon_errors)):
            logger.warning(f"Client {self.client_type}: recon_errors contains NaN or inf. Cleaning...")
            finite_errors = recon_errors[np.isfinite(recon_errors)]
            max_finite = np.max(finite_errors) if finite_errors.size > 0 else 100.0
            recon_errors = np.nan_to_num(recon_errors, nan=max_finite, posinf=max_finite, neginf=0.0)
        
        recon_errors = recon_errors / self.input_variance
        logger.info(f"Client {self.client_type}: recon_errors stats (fit) - min: {np.min(recon_errors):.4f}, max: {np.max(recon_errors):.4f}, mean: {np.mean(recon_errors):.4f}")
        recon_errors = np.clip(recon_errors, 0, 100.0)
        clipped_percentage = np.mean(recon_errors == 100.0) * 100
        logger.info(f"Client {self.client_type}: Percentage of recon_errors clipped to 100 (fit): {clipped_percentage:.2f}%")

        precisions, recalls, thresholds = precision_recall_curve(self.y_train, recon_errors)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        beta = 1.0
        f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        best_threshold = thresholds[np.argmax(f_beta_scores)]
        y_pred = (recon_errors > best_threshold).astype(int)
        acc = accuracy_score(self.y_train, y_pred)
        f1 = f1_score(self.y_train, y_pred, average='weighted')
        self.metrics["train_accuracy"].append(float(acc))
        self.metrics["train_f1_score"].append(float(f1))

        metrics = {"loss": loss, "accuracy": acc, "f1_score": f1, "cluster_id": config.get("cluster_id", 0)}
        logger.info(f"Client {self.client_type} fit metrics: {metrics}")
        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        logger.info(f"Client {self.client_type}: Received {len(parameters)} parameters for evaluation")
        expected_weights_len = len(self.vae.get_weights())
        has_all_weights = parameters[0][0] > 0.5
        logger.info(f"Client {self.client_type}: has_all_weights: {has_all_weights}, expected weights length: {expected_weights_len}")

        cluster_weights = parameters[1:expected_weights_len + 1]
        if len(cluster_weights) != expected_weights_len:
            logger.error(f"Client {self.client_type}: Expected {expected_weights_len} weights, but got {len(cluster_weights)}")
            raise ValueError(f"Expected {expected_weights_len} weights, but got {len(cluster_weights)}")

        dequantized_weights = []
        for w in cluster_weights:
            max_abs = np.max(np.abs(w))
            if max_abs == 0:
                dequantized_weights.append(w.astype(np.float32))
            else:
                dequantized_weights.append((w.astype(np.float32) * max_abs / 127))
        dequantized_weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in dequantized_weights]
        dequantized_weights = [np.clip(w, -10.0, 10.0) for w in dequantized_weights]
        logger.info(f"Client {self.client_type}: Dequantized weights length: {len(dequantized_weights)}")
        self.vae.set_weights(dequantized_weights)

        reconstructions = self.vae.predict(self.X_test, verbose=0)
        if np.any(np.isnan(reconstructions)) or np.any(np.isinf(reconstructions)):
            logger.warning(f"Client {self.client_type}: VAE predictions contain NaN or inf in evaluate. Cleaning...")
            reconstructions = np.nan_to_num(reconstructions, nan=0.0, posinf=0.0, neginf=0.0)
        
        reconstructions_rescaled = (reconstructions * 2.0) - 1.0
        recon_errors = np.mean(np.square(self.X_test - reconstructions_rescaled), axis=1)
        if np.any(np.isnan(recon_errors)) or np.any(np.isinf(recon_errors)):
            logger.warning(f"Client {self.client_type}: recon_errors contains NaN or inf in evaluate. Cleaning...")
            finite_errors = recon_errors[np.isfinite(recon_errors)]
            max_finite = np.max(finite_errors) if finite_errors.size > 0 else 100.0
            recon_errors = np.nan_to_num(recon_errors, nan=max_finite, posinf=max_finite, neginf=0.0)
        
        recon_errors = recon_errors / self.input_variance
        logger.info(f"Client {self.client_type}: recon_errors stats (evaluate) - min: {np.min(recon_errors):.4f}, max: {np.max(recon_errors):.4f}, mean: {np.mean(recon_errors):.4f}")
        recon_errors = np.clip(recon_errors, 0, 100.0)
        clipped_percentage = np.mean(recon_errors == 100.0) * 100
        logger.info(f"Client {self.client_type}: Percentage of recon_errors clipped to 100 (evaluate): {clipped_percentage:.2f}%")
        
        loss = float(np.mean(recon_errors))

        precisions, recalls, thresholds = precision_recall_curve(self.y_test, recon_errors)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        beta = 1.0
        f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        best_threshold = thresholds[np.argmax(f_beta_scores)]
        y_pred = (recon_errors > best_threshold).astype(int)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        if len(self.X_test) >= 30:
            subset_size = min(30, len(self.X_test))
            X_subset = self.X_test[:subset_size]
            def recon_error_predict(X):
                preds = self.vae.predict(X, verbose=0)
                if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                preds_rescaled = (preds * 2.0) - 1.0
                return np.mean(np.square(X - preds_rescaled), axis=1) / self.input_variance
            explainer = shap.KernelExplainer(recon_error_predict, X_subset)
            shap_values = explainer.shap_values(X_subset, nsamples=10, l1_reg="aic", eps=1e-5)
            feature_names = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            shap_summary = {name: float(np.mean(np.abs(shap_values[:, i]))) for i, name in enumerate(feature_names)}
            logger.info(f"Client {self.client_type}: SHAP feature importance: {shap_summary}")

        z_mean, z_log_var, z = self.encoder.predict(self.X_test, verbose=0)
        normal_mask = self.y_test == 0
        anomaly_mask = self.y_test == 1
        z_normal = z[normal_mask]
        z_anomaly = z[anomaly_mask]
        if len(z_normal) > 0:
            z_normal_mean = np.mean(z_normal, axis=0).tolist()
            z_normal_var = np.var(z_normal, axis=0).tolist()
            logger.info(f"Client {self.client_type}: Latent space (normal) - Mean: {z_normal_mean}, Variance: {z_normal_var}")
        if len(z_anomaly) > 0:
            z_anomaly_mean = np.mean(z_anomaly, axis=0).tolist()
            z_anomaly_var = np.var(z_anomaly, axis=0).tolist()
            logger.info(f"Client {self.client_type}: Latent space (anomaly) - Mean: {z_anomaly_mean}, Variance: {z_anomaly_var}")

        class_acc = {}
        class_f1 = {}
        for label in range(2):
            mask = self.y_test == label
            if mask.sum() > 0:
                class_acc[label] = float(accuracy_score(self.y_test[mask], y_pred[mask]))
                class_f1[label] = float(f1_score(self.y_test[mask], y_pred[mask], average='weighted'))
        logger.info(f"Client {self.client_type}: Class-wise accuracy: {class_acc}")
        logger.info(f"Client {self.client_type}: Class-wise F1-score: {class_f1}")

        maintenance_decisions = []
        for i, error in enumerate(recon_errors):
            state = self.q_agent.discretize_state(error)
            action = self.q_agent.choose_action(state)
            maintenance_decisions.append(action)
            actual_failure = self.y_test[i]
            if action == 1 and actual_failure == 1:
                reward = 1
            elif action == 1 and actual_failure == 0:
                reward = -1
            elif action == 0 and actual_failure == 1:
                reward = -2
            else:
                reward = 0
            next_state = self.q_agent.discretize_state(error)
            self.q_agent.update(state, action, reward, next_state)
        maintenance_acc = float(accuracy_score(self.y_test, maintenance_decisions))
        logger.info(f"Client {self.client_type}: Maintenance scheduling accuracy: {maintenance_acc:.4f}")

        cluster_losses = {}
        if has_all_weights:
            all_cluster_weights_flat = parameters[expected_weights_len + 1:]
            weights_per_cluster = expected_weights_len
            all_cluster_weights = [
                all_cluster_weights_flat[i:i + weights_per_cluster]
                for i in range(0, len(all_cluster_weights_flat), weights_per_cluster)
            ]
            logger.info(f"Client {self.client_type}: Processing {len(all_cluster_weights)} additional cluster weights")
            for cluster_idx, cluster_weights in enumerate(all_cluster_weights):
                if len(cluster_weights) != expected_weights_len:
                    logger.warning(f"Client {self.client_type}: Cluster {cluster_idx} weights length mismatch: expected {expected_weights_len}, got {len(cluster_weights)}")
                    continue
                dequantized_cluster_weights = []
                for w in cluster_weights:
                    max_abs = np.max(np.abs(w))
                    if max_abs == 0:
                        dequantized_cluster_weights.append(w.astype(np.float32))
                    else:
                        dequantized_cluster_weights.append((w.astype(np.float32) * max_abs / 127))
                dequantized_cluster_weights = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in dequantized_cluster_weights]
                dequantized_cluster_weights = [np.clip(w, -10.0, 10.0) for w in dequantized_cluster_weights]
                self.vae.set_weights(dequantized_cluster_weights)
                reconstructions = self.vae.predict(self.X_test, verbose=0)
                if np.any(np.isnan(reconstructions)) or np.any(np.isinf(reconstructions)):
                    reconstructions = np.nan_to_num(reconstructions, nan=0.0, posinf=0.0, neginf=0.0)
                reconstructions_rescaled = (reconstructions * 2.0) - 1.0
                cluster_recon_errors = np.mean(np.square(self.X_test - reconstructions_rescaled), axis=1)
                cluster_recon_errors = cluster_recon_errors / self.input_variance
                cluster_recon_errors = np.clip(cluster_recon_errors, 0, 100.0)
                cluster_loss = float(np.mean(cluster_recon_errors))
                cluster_losses[f"cluster_loss_{cluster_idx}"] = cluster_loss

        self.metrics["test_loss"].append(float(loss))
        self.metrics["test_accuracy"].append(float(acc))
        self.metrics["test_f1_score"].append(float(f1))
        self.metrics["maintenance_accuracy"].append(float(maintenance_acc))

        metrics = {
            "loss": float(loss),
            "accuracy": float(acc),
            "f1_score": float(f1),
            "maintenance_accuracy": float(maintenance_acc)
        }
        metrics.update(cluster_losses)
        logger.info(f"Client {self.client_type} evaluate metrics: {metrics}")
        return loss, len(self.X_test), metrics

    def save_metrics(self):
        with open(f"client_{self.client_type}_metrics.json", "w") as f:
            json.dump(self.metrics, f)

# Start the client
def main():
    import sys
    client_type = sys.argv[1] if len(sys.argv) > 1 else "L"
    client = FLClient(client_type)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    client.save_metrics()
    logger.info(f"Client {client_type} metrics saved to client_{client_type}_metrics.json")

if __name__ == "__main__":
    main()
