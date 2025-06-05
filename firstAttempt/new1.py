import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from typing import Dict, List, Tuple
import os

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Since the data is provided as text, we'll convert it to a DataFrame
    # In a real scenario, you'd load it from a file
    data = pd.read_csv('D:/FEDERATED LEARNING PROJECT/predictive_maintenance.csv')  # Replace with actual file path
    
    # Features and target
    features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = data[features]
    y = data['Target']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Define the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Client class
class MaintenanceClient(fl.client.NumPyClient):
    def __init__(self, cid: str, X_train, y_train, X_val, y_val):
        self.cid = cid
        self.model = create_model()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Training
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Get updated parameters and metrics
        updated_parameters = self.model.get_weights()
        num_examples = len(self.X_train)
        loss = history.history['loss'][-1]
        
        return updated_parameters, num_examples, {"loss": float(loss)}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}

# Custom q-FedAvg strategy
class QFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, q=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q  # q parameter for q-FedAvg

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        
        # Convert results to weights and compute differences
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Get global model weights
        global_weights = self.parameters
        
        # Calculate updates with q-FedAvg
        aggregated_updates = []
        total_weight = 0
        
        for client_weights, num_examples in weights_results:
            # Calculate update (difference)
            update = [
                (gw - cw) * (num_examples ** self.q)
                for gw, cw in zip(global_weights, client_weights)
            ]
            
            if not aggregated_updates:
                aggregated_updates = update
            else:
                aggregated_updates = [
                    agg + upd for agg, upd in zip(aggregated_updates, update)
                ]
            total_weight += num_examples ** self.q
        
        # Average the updates
        aggregated_updates = [
            upd / total_weight for upd in aggregated_updates
        ]
        
        # Apply updates to global weights
        aggregated_weights = [
            gw - upd for gw, upd in zip(global_weights, aggregated_updates)
        ]
        
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

# Function to create clients
def create_clients(X_train, y_train, X_test, y_test, num_clients=3):
    # Split data among clients
    X_train_splits = np.array_split(X_train, num_clients)
    y_train_splits = np.array_split(y_train, num_clients)
    X_test_splits = np.array_split(X_test, num_clients)
    y_test_splits = np.array_split(y_test, num_clients)
    
    clients = []
    for i in range(num_clients):
        client = MaintenanceClient(
            f"client_{i}",
            X_train_splits[i],
            y_train_splits[i],
            X_test_splits[i],
            y_test_splits[i]
        )
        clients.append(client)
    return clients

# Client function for Flower
def client_fn(cid: str) -> fl.client.Client:
    client_idx = int(cid.split('_')[1])
    return clients[client_idx]

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Create clients
    clients = create_clients(X_train, y_train, X_test, y_test, num_clients=3)
    
    # Define FedAvg strategy
    fedavg_strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    
    # Define q-FedAvg strategy
    qfedavg_strategy = QFedAvg(
        q=1.0,  # Adjust q parameter as needed
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    
    # Function to start server with a given strategy
    def start_server(strategy, strategy_name):
        print(f"\nStarting simulation with {strategy_name}")
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy
        )
    
    # Start FedAvg simulation
    start_server(fedavg_strategy, "FedAvg")
    
    # Start q-FedAvg simulation
    start_server(qfedavg_strategy, "q-FedAvg")
    
    # Start client processes (in practice, this would be on separate machines)
    for client in clients:
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client
        )
