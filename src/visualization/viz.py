import matplotlib.pyplot as plt
import os
import numpy as np

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def plot_bins(config):
    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"
    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")
    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    plt.hist(train_scores, bins=100, alpha=0.5, label="actor1_w1")
    plt.hist(actor2_scores, bins=100, alpha=0.5, label="actor2")
    plt.legend()

    # plots folder
    plots_folder = f"{eval_results_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f"{plots_folder}/bins.png", dpi=300, bbox_inches="tight")

    plt.show()
    


def plot_anomaly_scores_distribution(config):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"

    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # Select score type
    score_type = config["evaluation"].get("score_type", "combined")

    actor1_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    # Threshold computed from Actor 1 (normal behavior)
    threshold_percentile = config["evaluation"]["threshold_percentile"]
    threshold = np.percentile(actor1_scores, threshold_percentile)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 4),
        sharey=True
    )

    # --- Actor 1 ---
    ax1.plot(actor1_scores, linewidth=1)
    ax1.axhline(threshold, linestyle="--", color="red")
    ax1.set_title("Actor 1 (Train / Normal)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Anomaly Score")
    ax1.grid(alpha=0.3)

    # --- Actor 2 ---
    ax2.plot(actor2_scores, linewidth=1)
    ax2.axhline(threshold, linestyle="--", color="red")
    ax2.set_title("Actor 2 (Test / Possibly Anomalous)")
    ax2.set_xlabel("Time Step")
    ax2.grid(alpha=0.3)

    # Global title
    fig.suptitle("Anomaly Score Distributions per Actor", fontsize=14)

    plt.tight_layout()

    # Save plots folder
    plots_folder = f"{eval_results_folder}/plots"
    os.makedirs(plots_folder, exist_ok=True)

    save_path = f"{plots_folder}/anomaly_scores_side_by_side.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_boxplot(config):
    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"
    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")
    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    plt.figure(figsize=(8, 5))

    data = [
        train_scores,
        actor2_scores,
    ]

    plt.boxplot(data,
                showfliers=False)

    plt.title("Score Distribution Comparison")
    plt.ylabel("Score")

    #
    # plots folder
    plots_folder = f"{eval_results_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f"{plots_folder}/boxplot.png", dpi=300, bbox_inches="tight")

    plt.show()


