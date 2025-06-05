import json
import matplotlib.pyplot as plt
import os

# Function to load metrics with error handling
def load_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load aggregated metrics
aggregated_metrics = load_metrics("aggregated_metrics.json")
if aggregated_metrics is None or not aggregated_metrics.get("loss") or not aggregated_metrics.get("accuracy"):
    print("Warning: Aggregated metrics are empty or not loaded. Generating empty plot.")
    aggregated_metrics = {"loss": [], "accuracy": []}

# Load client metrics
client_types = ["L", "M", "H"]
client_metrics = {}
for ct in client_types:
    metrics = load_metrics(f"client_{ct}_metrics.json")
    if metrics is None or not any(metrics.get(key) for key in ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]):
        print(f"Warning: Metrics for client {ct} are empty or not loaded. Using empty lists.")
        client_metrics[ct] = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}
    else:
        client_metrics[ct] = metrics

# Plot Aggregated Metrics
plt.figure(figsize=(12, 5))

# Aggregated Loss
plt.subplot(1, 2, 1)
if aggregated_metrics["loss"]:
    plt.plot(range(1, len(aggregated_metrics["loss"]) + 1), aggregated_metrics["loss"], label="Aggregated Loss", marker="o")
    plt.title("Aggregated Loss Across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Loss Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Aggregated Loss Across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")

# Aggregated Accuracy
plt.subplot(1, 2, 2)
if aggregated_metrics["accuracy"]:
    plt.plot(range(1, len(aggregated_metrics["accuracy"]) + 1), aggregated_metrics["accuracy"], label="Aggregated Accuracy", marker="o")
    plt.title("Aggregated Accuracy Across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Accuracy Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Aggregated Accuracy Across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("aggregated_metrics.png")
plt.close()

# Plot Per-Client Metrics
plt.figure(figsize=(12, 10))

# Client Training Loss
plt.subplot(2, 2, 1)
has_data = False
for ct in client_types:
    if client_metrics[ct]["train_loss"]:
        plt.plot(range(1, len(client_metrics[ct]["train_loss"]) + 1), client_metrics[ct]["train_loss"], label=f"Client {ct}", marker="o")
        has_data = True
if has_data:
    plt.title("Training Loss Per Client")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Training Loss Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Training Loss Per Client")
    plt.xlabel("Round")
    plt.ylabel("Loss")

# Client Training Accuracy
plt.subplot(2, 2, 2)
has_data = False
for ct in client_types:
    if client_metrics[ct]["train_accuracy"]:
        plt.plot(range(1, len(client_metrics[ct]["train_accuracy"]) + 1), client_metrics[ct]["train_accuracy"], label=f"Client {ct}", marker="o")
        has_data = True
if has_data:
    plt.title("Training Accuracy Per Client")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Training Accuracy Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Training Accuracy Per Client")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

# Client Test Loss
plt.subplot(2, 2, 3)
has_data = False
for ct in client_types:
    if client_metrics[ct]["test_loss"]:
        plt.plot(range(1, len(client_metrics[ct]["test_loss"]) + 1), client_metrics[ct]["test_loss"], label=f"Client {ct}", marker="o")
        has_data = True
if has_data:
    plt.title("Test Loss Per Client")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Test Loss Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Test Loss Per Client")
    plt.xlabel("Round")
    plt.ylabel("Loss")

# Client Test Accuracy
plt.subplot(2, 2, 4)
has_data = False
for ct in client_types:
    if client_metrics[ct]["test_accuracy"]:
        plt.plot(range(1, len(client_metrics[ct]["test_accuracy"]) + 1), client_metrics[ct]["test_accuracy"], label=f"Client {ct}", marker="o")
        has_data = True
if has_data:
    plt.title("Test Accuracy Per Client")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
else:
    plt.text(0.5, 0.5, "No Test Accuracy Data", horizontalalignment='center', verticalalignment='center')
    plt.title("Test Accuracy Per Client")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("client_metrics.png")
plt.close()

print("Visualizations saved as 'aggregated_metrics.png' and 'client_metrics.png'")
