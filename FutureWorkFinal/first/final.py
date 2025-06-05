import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths to the saved metrics files
SERVER_METRICS_PATH = "C:/Users/rkjra/Desktop/FL/IFCA/metrics_history.npy"
CLIENT_METRICS_PATHS = {
    'L': "C:/Users/rkjra/Desktop/FL/IFCA/client_L_metrics.npy",
    'M': "C:/Users/rkjra/Desktop/FL/IFCA/client_M_metrics.npy",
    'H': "C:/Users/rkjra/Desktop/FL/IFCA/client_H_metrics.npy"
}

# Load server metrics
server_metrics = np.load(SERVER_METRICS_PATH, allow_pickle=True).item()
server_rounds = [entry[0] for entry in server_metrics['loss']]
server_loss = [entry[1] for entry in server_metrics['loss']]
server_accuracy = [entry[1] for entry in server_metrics['accuracy']]
server_f1_score = [entry[1] for entry in server_metrics['f1_score']]

# Load client metrics
client_metrics = {}
for device, path in CLIENT_METRICS_PATHS.items():
    if os.path.exists(path):
        metrics = np.load(path, allow_pickle=True).item()
        client_metrics[device] = {
            'rounds': [entry[0] for entry in metrics['loss']],
            'loss': [entry[1] for entry in metrics['loss']],
            'accuracy': [entry[1] for entry in metrics['accuracy']],
            'f1_score': [entry[1] for entry in metrics['f1_score']]
        }
    else:
        print(f"Metrics file for client {device} not found at {path}")

# Create a directory for saving plots
OUTPUT_DIR = "C:/Users/rkjra/Desktop/FL/IFCA/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot 1: Server Metrics Over Rounds
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(server_rounds, server_loss, marker='o', color='b', label='Server Loss')
plt.title('Server Loss Over Rounds')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(server_rounds, server_accuracy, marker='o', color='g', label='Server Accuracy')
plt.title('Server Accuracy Over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(server_rounds, server_f1_score, marker='o', color='r', label='Server F1-Score')
plt.title('Server F1-Score Over Rounds')
plt.xlabel('Round')
plt.ylabel('F1-Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'server_metrics.png'))
plt.close()

# Plot 2: Client Metrics Comparison
plt.figure(figsize=(12, 8))

# Loss
plt.subplot(3, 1, 1)
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['loss'], marker='o', label=f'Client {device} Loss')
plt.title('Client Loss Over Rounds')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy
plt.subplot(3, 1, 2)
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['accuracy'], marker='o', label=f'Client {device} Accuracy')
plt.title('Client Accuracy Over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# F1-Score
plt.subplot(3, 1, 3)
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['f1_score'], marker='o', label=f'Client {device} F1-Score')
plt.title('Client F1-Score Over Rounds')
plt.xlabel('Round')
plt.ylabel('F1-Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'client_metrics_comparison.png'))
plt.close()

# Plot 3: Combined Server and Client Metrics
plt.figure(figsize=(12, 8))

# Loss
plt.subplot(3, 1, 1)
plt.plot(server_rounds, server_loss, marker='o', color='b', label='Server Loss', linewidth=2, linestyle='--')
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['loss'], marker='o', label=f'Client {device} Loss')
plt.title('Loss Over Rounds (Server vs Clients)')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy
plt.subplot(3, 1, 2)
plt.plot(server_rounds, server_accuracy, marker='o', color='g', label='Server Accuracy', linewidth=2, linestyle='--')
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['accuracy'], marker='o', label=f'Client {device} Accuracy')
plt.title('Accuracy Over Rounds (Server vs Clients)')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# F1-Score
plt.subplot(3, 1, 3)
plt.plot(server_rounds, server_f1_score, marker='o', color='r', label='Server F1-Score', linewidth=2, linestyle='--')
for device, metrics in client_metrics.items():
    plt.plot(metrics['rounds'], metrics['f1_score'], marker='o', label=f'Client {device} F1-Score')
plt.title('F1-Score Over Rounds (Server vs Clients)')
plt.xlabel('Round')
plt.ylabel('F1-Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_metrics.png'))
plt.close()

print(f"Plots saved in {OUTPUT_DIR}")
