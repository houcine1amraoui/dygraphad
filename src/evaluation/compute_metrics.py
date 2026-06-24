import numpy as np
import os
import yaml
import numpy as np

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def compute_point_wise_metrics(config):
    """
    Detection rate (Actor2 test) → expect HIGH
    False positive rate (Actor 1 train) → expect LOW
    """

    # choose which score to use
    dataset_name = config["preprocessing"]["dataset_name"]
    
    print(f"Computing point-wise metrics for {config['evaluation'].get('model', 'unknown')} on {dataset_name}")

    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"

    scores = np.load(scores_path, allow_pickle=True)
    

    train_scores = scores["train"]
    actor2_scores = scores["actor2_test"]

    # threshold from NORMAL data only
    threshold_percentile = config["evaluation"]["threshold_percentile"]

    threshold = np.percentile(train_scores, threshold_percentile)
    
    # metrics
    detection_rate = np.mean(actor2_scores > threshold)
    false_positive_rate = np.mean(train_scores > threshold)

    # save properly
    metrics = {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
    }
    print(metrics)

    with open(os.path.join(eval_results_folder, "point_wise_metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)

    return metrics

def compute_segment_wise_metrics(config):
    
    print(f"Computing segment-wise metrics for {config['evaluation'].get('model', 'unknown')}...")

    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"

    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")

    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    # threshold from NORMAL data only
    threshold_percentile = config["evaluation"]["threshold_percentile"]
    threshold = np.percentile(train_scores, threshold_percentile)

    def extract_segments(binary_seq):
        segments = []
        start = None

        for i, val in enumerate(binary_seq):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i - 1))
                start = None

        if start is not None:
            segments.append((start, len(binary_seq) - 1))

        return segments

    # --- Actor2 (anomalous) ---
    pred_actor2 = actor2_scores > threshold
    seg_actor2 = extract_segments(pred_actor2)
    print("Anomalous segments detected in Actor2 test set:", seg_actor2)
    print(len(seg_actor2), "anomalous segments detected in Actor2 test set.")

    detection_rate = 1.0 if len(seg_actor2) > 0 else 0.0
    
    coverage = np.mean(pred_actor2)
    detection_delay = np.argmax(pred_actor2) if np.any(pred_actor2) else -1

    # --- Actor1 (normal) ---
    pred_actor1 = train_scores > threshold
    seg_actor1 = extract_segments(pred_actor1)

    false_positive_rate = len(seg_actor1) / len(pred_actor1)

    
    metrics = {
        "detection_rate": detection_rate,
        "coverage": coverage,
        "detection_delay": detection_delay,
        "false_positive_rate": false_positive_rate,
    }
    print(metrics)

    with open(os.path.join(eval_results_folder, "segment_metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)

    return metrics