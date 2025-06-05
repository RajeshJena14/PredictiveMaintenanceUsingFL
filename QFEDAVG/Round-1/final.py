import numpy as np
import matplotlib.pyplot as plt
import os

# Configure plot style
plt.style.use('seaborn-v0_8')

def load_client_metrics(metrics_file, client_name):
    """Load client metrics from .npy file."""
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Client {client_name} metrics file {metrics_file} not found.")
    metrics_history = np.load(metrics_file, allow_pickle=True).item()
    return metrics_history

def load_server_metrics(metrics_file):
    """Load server-side aggregated metrics from metrics_history.npy."""
    metrics_history = np.load(metrics_file, allow_pickle=True).item()
    rounds = sorted(set(round_num for round_num, _ in metrics_history['loss']))
    server_metrics = {
        'loss': [value for round_num, value in metrics_history['loss'] if round_num in rounds],
        'accuracy': [value for round_num, value in metrics_history['accuracy'] if round_num in rounds],
        'f1_score': [value for round_num, value in metrics_history['f1_score'] if round_num in rounds]
    }
    return rounds, server_metrics

def plot_metrics(rounds, client_metrics, server_metrics, output_file):
    """Plot per-client and overall metrics."""
    clients = list(client_metrics.keys())
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Colors for clients and server
    colors = {'L': 'blue', 'M': 'green', 'H': 'red', 'Overall': 'black'}
    linestyles = {'L': '-', 'M': '--', 'H': '-.', 'Overall': '-'}

    # Plot Loss
    for client in clients:
        client_rounds = [r for r, _ in client_metrics[client]['loss']]
        client_values = [v for _, v in client_metrics[client]['loss']]
        ax1.plot(client_rounds, client_values, 
                 label=f'Client {client}', color=colors[client], linestyle=linestyles[client])
    ax1.plot(rounds, server_metrics['loss'], label='Overall', color=colors['Overall'], 
             linestyle=linestyles['Overall'], linewidth=2)
    ax1.set_title('Loss per Round')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    for client in clients:
        client_rounds = [r for r, _ in client_metrics[client]['accuracy']]
        client_values = [v for _, v in client_metrics[client]['accuracy']]
        ax2.plot(client_rounds, client_values, 
                 label=f'Client {client}', color=colors[client], linestyle=linestyles[client])
    ax2.plot(rounds, server_metrics['accuracy'], label='Overall', color=colors['Overall'], 
             linestyle=linestyles['Overall'], linewidth=2)
    ax2.set_title('Accuracy per Round')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Plot F1-Score
    for client in clients:
        client_rounds = [r for r, _ in client_metrics[client]['f1_score']]
        client_values = [v for _, v in client_metrics[client]['f1_score']]
        ax3.plot(client_rounds, client_values, 
                 label=f'Client {client}', color=colors[client], linestyle=linestyles[client])
    ax3.plot(rounds, server_metrics['f1_score'], label='Overall', color=colors['Overall'], 
             linestyle=linestyles['Overall'], linewidth=2)
    ax3.set_title('F1-Score per Round')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def main():
    """Main function to load and visualize metrics."""
    # File paths
    base_dir = "./"
    metrics_file = os.path.join(base_dir, "metrics_history.npy")
    client_metrics_files = {
        'L': os.path.join(base_dir, "client_L_metrics.npy"),
        'M': os.path.join(base_dir, "client_M_metrics.npy"),
        'H': os.path.join(base_dir, "client_H_metrics.npy")
    }
    output_file = os.path.join(base_dir, "metrics_visualization.png")

    # Verify server metrics file exists
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Server metrics file {metrics_file} not found.")

    # Load server metrics
    rounds, server_metrics = load_server_metrics(metrics_file)

    # Load client metrics
    client_metrics = {}
    for client, metrics_file in client_metrics_files.items():
        try:
            client_metrics[client] = load_client_metrics(metrics_file, client)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping client {client}.")
            continue

    if not client_metrics:
        raise ValueError("No client metrics files found. Please ensure at least one client metrics file exists.")

    # Plot metrics
    plot_metrics(rounds, client_metrics, server_metrics, output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
