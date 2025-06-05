import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metrics_history(file_path="metrics_history.npy"):
    """Load the aggregated metrics history."""
    try:
        metrics_history = np.load(file_path, allow_pickle=True).item()
        logger.info("Successfully loaded metrics_history.npy")
        return metrics_history
    except FileNotFoundError:
        logger.error("❌ metrics_history.npy not found.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading metrics_history.npy: {e}")
        return None

def load_client_metrics_history(file_path="client_metrics_history.npy"):
    """Load the per-client metrics history."""
    try:
        client_metrics_history = np.load(file_path, allow_pickle=True).item()
        logger.info("Successfully loaded client_metrics_history.npy")
        return client_metrics_history
    except FileNotFoundError:
        logger.error("❌ client_metrics_history.npy not found. Per-client metrics unavailable.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading client_metrics_history.npy: {e}")
        return None

def plot_metrics(metrics_history, client_metrics_history):
    """Plot aggregated and per-device (averaged) metrics trends across rounds."""
    if metrics_history is None:
        return

    rounds = [entry[0] for entry in metrics_history["loss"]]
    losses = [entry[1] for entry in metrics_history["loss"]]
    accuracies = [entry[1] for entry in metrics_history["accuracy"]]
    f1_scores = [entry[1] for entry in metrics_history["f1_score"]]

    plt.figure(figsize=(15, 10))

    # Plot Aggregated Metrics
    plt.subplot(2, 3, 1)
    plt.plot(rounds, losses, marker='o', color='r', label='Aggregated Loss')
    plt.title('Aggregated Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(rounds, accuracies, marker='o', color='g', label='Aggregated Accuracy')
    plt.title('Aggregated Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(rounds, f1_scores, marker='o', color='b', label='Aggregated F1-Score')
    plt.title('Aggregated F1-Score per Round')
    plt.xlabel('Round')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()

    # Plot Per-Device Metrics (averaged across clients)
    if client_metrics_history:
        devices = ['L', 'M', 'H']
        colors = {'L': 'orange', 'M': 'purple', 'H': 'cyan'}
        
        # Average per-device loss
        plt.subplot(2, 3, 4)
        for dev in devices:
            if dev in client_metrics_history["loss"]:
                dev_losses = [entry[1] for entry in client_metrics_history["loss"][dev]]
                avg_losses = [np.mean(dev_losses[i::len(devices)]) for i in range(len(rounds))]  # Average per round
                plt.plot(rounds, avg_losses, marker='o', color=colors[dev], label=f'{dev} Avg Loss')
        plt.title('Per-Device Average Loss per Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Average per-device accuracy
        plt.subplot(2, 3, 5)
        for dev in devices:
            if dev in client_metrics_history["accuracy"]:
                dev_accs = [entry[1] for entry in client_metrics_history["accuracy"][dev]]
                avg_accs = [np.mean(dev_accs[i::len(devices)]) for i in range(len(rounds))]  # Average per round
                plt.plot(rounds, avg_accs, marker='o', color=colors[dev], label=f'{dev} Avg Accuracy')
        plt.title('Per-Device Average Accuracy per Round')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        # Average per-device F1-score
        plt.subplot(2, 3, 6)
        for dev in devices:
            if dev in client_metrics_history["f1_score"]:
                dev_f1s = [entry[1] for entry in client_metrics_history["f1_score"][dev]]
                avg_f1s = [np.mean(dev_f1s[i::len(devices)]) for i in range(len(rounds))]  # Average per round
                plt.plot(rounds, avg_f1s, marker='o', color=colors[dev], label=f'{dev} Avg F1-Score')
        plt.title('Per-Device Average F1-Score per Round')
        plt.xlabel('Round')
        plt.ylabel('F1-Score')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig("performance_trends_per_device.png")
    logger.info("Performance trends plot saved as 'performance_trends_per_device.png'")
    plt.show()

def print_per_round_metrics(metrics_history, client_metrics_history):
    """Print aggregated and per-device (averaged) metrics per round."""
    if metrics_history is None:
        return

    rounds = [entry[0] for entry in metrics_history["loss"]]
    losses = [entry[1] for entry in metrics_history["loss"]]
    accuracies = [entry[1] for entry in metrics_history["accuracy"]]
    f1_scores = [entry[1] for entry in metrics_history["f1_score"]]

    # Aggregated Metrics
    logger.info("\n=== Aggregated Per-Round Performance ===")
    logger.info(f"{'Round':<6} {'Loss':<12} {'Accuracy':<12} {'F1-Score':<12}")
    logger.info("-" * 42)
    for r, l, a, f in zip(rounds, losses, accuracies, f1_scores):
        logger.info(f"{r:<6} {l:<12.4f} {a:<12.4f} {f:<12.4f}")
    logger.info("-" * 42)

    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    logger.info("\n=== Aggregated Overall Performance ===")
    logger.info(f"Average Loss:     {avg_loss:.4f}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average F1-Score: {avg_f1:.4f}")

    # Per-Device Metrics (averaged)
    if client_metrics_history:
        devices = ['L', 'M', 'H']
        for dev in devices:
            if dev in client_metrics_history["loss"]:
                logger.info(f"\n=== Per-Round Performance for Device {dev} (Averaged) ===")
                logger.info(f"{'Round':<6} {'Loss':<12} {'Accuracy':<12} {'F1-Score':<12}")
                logger.info("-" * 42)
                dev_losses = [entry[1] for entry in client_metrics_history["loss"][dev]]
                dev_accs = [entry[1] for entry in client_metrics_history["accuracy"][dev]]
                dev_f1s = [entry[1] for entry in client_metrics_history["f1_score"][dev]]
                for r, l, a, f in zip(rounds, dev_losses[::len(devices)], dev_accs[::len(devices)], dev_f1s[::len(devices)]):
                    logger.info(f"{r:<6} {l:<12.4f} {a:<12.4f} {f:<12.4f}")
                logger.info("-" * 42)
                logger.info(f"Average Loss for {dev}:     {np.mean(dev_losses[::len(devices)]):.4f}")
                logger.info(f"Average Accuracy for {dev}: {np.mean(dev_accs[::len(devices)]):.4f}")
                logger.info(f"Average F1-Score for {dev}: {np.mean(dev_f1s[::len(devices)]):.4f}")

def visualize_performance():
    """Main function to visualize and review performance."""
    metrics_history = load_metrics_history()
    client_metrics_history = load_client_metrics_history()
    if metrics_history:
        plot_metrics(metrics_history, client_metrics_history)
        print_per_round_metrics(metrics_history, client_metrics_history)

if __name__ == "__main__":
    visualize_performance()
