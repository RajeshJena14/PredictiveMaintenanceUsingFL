import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import json

# Load and preprocess the dataset
def load_data(client_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv")
    df_client = df[df["Type"] == client_type]
    X = df_client[["Air temperature [K]", "Process temperature [K]", 
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].values
    y = df_client["Target"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the model
def create_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(5,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Flower client class
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_type: str):
        self.client_type = client_type
        self.X_train, self.X_test, self.y_train, self.y_test = load_data(client_type)
        self.model = create_model()
        self.metrics = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    def get_parameters(self, config):
        return [val.numpy() for val in self.model.trainable_weights]

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        # Log training metrics
        ''' self.metrics["train_loss"].append(float(history.history["loss"][-1]))
         self.metrics["train_accuracy"].append(float(history.history["accuracy"][-1]))
         return self.get_parameters(config), len(self.X_train), {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1])
         }  '''
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1])
        }
        print(f"Client {self.client_type} returning metrics: {metrics}")
        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        # Log evaluation metrics
        self.metrics["test_loss"].append(loss)
        self.metrics["test_accuracy"].append(accuracy)
        return loss, len(self.X_test), {"loss": loss, "accuracy": accuracy}

    def save_metrics(self):
        with open(f"client_{self.client_type}_metrics.json", "w") as f:
            json.dump(self.metrics, f)

# Start the client
def main():
    import sys
    client_type = sys.argv[1] if len(sys.argv) > 1 else "L"
    client = FLClient(client_type)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    client.save_metrics()  # Save metrics after training
    print(f"Client {client_type} metrics saved to client_{client_type}_metrics.json")

if __name__ == "__main__":
    main()
