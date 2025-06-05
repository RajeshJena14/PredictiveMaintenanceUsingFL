import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import cm

# Parse arguments for base directory
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/rkjra/Desktop/FL/FeSEM/", help='Base directory for metrics files')
args = parser.parse_args()
base_dir = args.base_dir

# Define client labels
client_labels = ['L', 'M', 'H']

# Dynamic colors and line styles
num_clients = len(client_labels)
colors = cm.tab10(np.linspace(0, 1, num_clients))
linestyles = ['-', '--', '-.', ':'][:num_clients]
markers = ['o', 's', 'D'][:num_clients]

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
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("Federated Learning Metrics Comparison (Enhanced FeSEM)", fontsize=18)
    
    metrics = ['loss', 'accuracy', 'f1_score']
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_title(f"Validation {metric.capitalize()}", fontsize=14)
        
        # Plot clients
        for j, client in enumerate(client_labels):
            if client in client_metrics and metric in client_metrics[client]:
                rounds, values = zip(*client_metrics[client][metric])
                valid_pairs = [(r, v) for r, v in zip(rounds, values) if not np.isnan(v) and not np.isinf(v)]
                if valid_pairs:
                    valid_rounds, valid_values = zip(*valid_pairs)
                    ax.plot(valid_rounds, valid_values, label=client, color=colors[j], linestyle=linestyles[j], marker=markers[j], markersize=6)
        
        # Plot server
        if metric in server_metrics and server_metrics[metric]:
            server_rounds, server_values = zip(*server_metrics[metric])
            valid_pairs = [(r, v) for r, v in zip(server_rounds, server_values) if not np.isnan(v) and not np.isinf(v)]
            if valid_pairs:
                valid_rounds, valid_values = zip(*valid_pairs)
                ax.plot(valid_rounds, valid_values, label='Overall', color='black', linewidth=2.5, marker='*', markersize=8)
        
        ax.set_ylabel(f"Validation {metric.capitalize()}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    axes[-1].set_xlabel("Round", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(base_dir, "metrics_visualization_fesem.png"), dpi=300)
    plt.show()

def main():
    # Load data
    client_metrics = load_client_metrics(base_dir, client_labels)
    server_metrics = load_server_metrics(base_dir)
    
    # Plot
    plot_metrics(client_metrics, server_metrics, client_labels)

if __name__ == "__main__":
    main()
