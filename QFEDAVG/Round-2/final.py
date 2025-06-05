import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import cm

# Parse arguments for base directory
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/rkjra/Desktop/FL/QFEDAVG/", help='Base directory for metrics files')
args = parser.parse_args()
base_dir = args.base_dir

# Define client labels
client_labels = ['L', 'M', 'H']

# Dynamic colors and line styles
num_clients = len(client_labels)
colors = cm.tab10(np.linspace(0, 1, num_clients))
linestyles = ['-', '--', '-.', ':'][:num_clients]

def load_client_metrics(base_dir, client_labels):
    client_metrics = {}
    for client in client_labels:
        file_path = os.path.join(base_dir, f"client_{client}_metrics.npy")
        if os.path.exists(file_path):
            client_metrics[client] = np.load(file_path, allow_pickle=True).item()
        else:
            print(f"Warning: Metrics file for client {client} not found.")
    return client_metrics

def load_server_metrics(base_dir):
    server_metrics_file = os.path.join(base_dir, "metrics_history.npy")
    if os.path.exists(server_metrics_file):
        return np.load(server_metrics_file, allow_pickle=True).item()
    else:
        raise FileNotFoundError("Server metrics file not found.")

def plot_metrics(client_metrics, server_metrics, client_labels):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle("Federated Learning Metrics Comparison", fontsize=16)
    
    metrics = ['loss', 'accuracy', 'f1_score']
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_title(f"Validation {metric.capitalize()}")
        
        # Plot clients
        for j, client in enumerate(client_labels):
            if client in client_metrics and metric in client_metrics[client]:
                rounds, values = zip(*client_metrics[client][metric])
                ax.plot(rounds, values, label=client, color=colors[j], linestyle=linestyles[j])
        
        # Plot server
        if metric in server_metrics:
            server_rounds, server_values = zip(*server_metrics[metric])
            ax.plot(server_rounds, server_values, label='Overall', color='black', linewidth=2)
        
        ax.set_ylabel(f"Validation {metric.capitalize()}")
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel("Round")
    fig.tight_layout()
    plt.savefig(os.path.join(base_dir, "metrics_visualization.png"))
    plt.show()

def main():
    # Load data
    client_metrics = load_client_metrics(base_dir, client_labels)
    server_metrics = load_server_metrics(base_dir)
    
    # Plot
    plot_metrics(client_metrics, server_metrics, client_labels)

if __name__ == "__main__":
    main()
