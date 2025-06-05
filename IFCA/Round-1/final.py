import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

# Directory where metrics are saved
METRICS_DIR = "C:/Users/rkjra/Desktop/FL/IFCA/"
DEVICES = ['L', 'M', 'H']

def load_metrics(file_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load metrics from a .npy file.
    Returns a dictionary with keys 'loss', 'accuracy', 'f1_score' and values as lists of (round, value).
    """
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {e}")
        return {"loss": [], "accuracy": [], "f1_score": []}

def plot_server_metrics(metrics: Dict[str, List[Tuple[int, float]]], save_path: str) -> None:
    """
    Plot server-side aggregated metrics (loss, accuracy, F1-score) over rounds.
    """
    # Extract rounds and values
    rounds_loss = [r for r, _ in metrics['loss']]
    losses = [v for _, v in metrics['loss']]
    rounds_acc = [r for r, _ in metrics['accuracy']]
    accuracies = [v for _, v in metrics['accuracy']]
    rounds_f1 = [r for r, _ in metrics['f1_score']]
    f1_scores = [v for _, v in metrics['f1_score']]

    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Loss
    ax1.plot(rounds_loss, losses, marker='o', color='red', label='Loss')
    ax1.set_title('Server Aggregated Loss Over Rounds')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(rounds_acc, accuracies, marker='o', color='blue', label='Accuracy')
    ax2.set_title('Server Aggregated Accuracy Over Rounds')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    # Plot F1-Score
    ax3.plot(rounds_f1, f1_scores, marker='o', color='green', label='F1-Score')
    ax3.set_title('Server Aggregated F1-Score Over Rounds')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('F1-Score')
    ax3.grid(True)
    ax3.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Server metrics plot saved to {save_path}")

def plot_client_metrics(device_metrics: Dict[str, Dict[str, List[Tuple[int, float]]]], save_path: str) -> None:
    """
    Plot client-side metrics (loss, accuracy, F1-score) over rounds for all devices.
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Colors for different devices
    colors = {'L': 'red', 'M': 'blue', 'H': 'green'}

    # Plot Loss for each device
    for device, metrics in device_metrics.items():
        rounds = [r for r, _ in metrics['loss']]
        values = [v for _, v in metrics['loss']]
        ax1.plot(rounds, values, marker='o', color=colors[device], label=f'Client {device}')
    ax1.set_title('Client Loss Over Rounds')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot Accuracy for each device
    for device, metrics in device_metrics.items():
        rounds = [r for r, _ in metrics['accuracy']]
        values = [v for _, v in metrics['accuracy']]
        ax2.plot(rounds, values, marker='o', color=colors[device], label=f'Client {device}')
    ax2.set_title('Client Accuracy Over Rounds')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    # Plot F1-Score for each device
    for device, metrics in device_metrics.items():
        rounds = [r for r, _ in metrics['f1_score']]
        values = [v for _, v in metrics['f1_score']]
        ax3.plot(rounds, values, marker='o', color=colors[device], label=f'Client {device}')
    ax3.set_title('Client F1-Score Over Rounds')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('F1-Score')
    ax3.grid(True)
    ax3.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Client metrics plot saved to {save_path}")

def main():
    # Load server metrics
    server_metrics_path = os.path.join(METRICS_DIR, "metrics_history.npy")
    server_metrics = load_metrics(server_metrics_path)

    # Load client metrics for each device
    device_metrics = {}
    for device in DEVICES:
        client_metrics_path = os.path.join(METRICS_DIR, f"client_{device}_metrics.npy")
        device_metrics[device] = load_metrics(client_metrics_path)

    # Plot server metrics
    server_plot_path = os.path.join(METRICS_DIR, "server_metrics_plot.png")
    plot_server_metrics(server_metrics, server_plot_path)

    # Plot client metrics
    client_plot_path = os.path.join(METRICS_DIR, "client_metrics_plot.png")
    plot_client_metrics(device_metrics, client_plot_path)

if __name__ == "__main__":
    main()
