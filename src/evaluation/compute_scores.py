import numpy as np

from src.utils.get_folders_utils import get_evaluation_results_main_folder

import numpy as np

def compute_scores(config):
    """
    Compute final timestamp anomaly scores from normalized
    device-level anomaly scores.

    Input:
        normalized_scores.npz

        train        -> (T,N)
        val          -> (T,N)
        actor2_test  -> (T,N)
        actor1_test  -> (T,N)

    Output:
        scores.npz

        train        -> (T,)
        val          -> (T,)
        actor2_test  -> (T,)
        actor1_test  -> (T,)
    """

    print("Computing anomaly scores...")

    eval_results_folder = get_evaluation_results_main_folder(config)

    data = np.load(
        f"{eval_results_folder}/norm_errors.npz"
    )

    aggregation = config["evaluation"].get(
        "score_aggregation",
        "mean"
    )

    scores = {}

    for split in [
        "train",
        "val",
        "actor2_test",
        "actor1_test"
    ]:

        device_scores = data[split]  # (T,N)

        if aggregation == "mean":

            timestamp_scores = np.mean(
                device_scores,
                axis=1
            )

        elif aggregation == "max":

            timestamp_scores = np.max(
                device_scores,
                axis=1
            )

        elif aggregation == "sum":

            timestamp_scores = np.sum(
                device_scores,
                axis=1
            )

        elif aggregation == "mean_max":

            timestamp_scores = (
                0.5 * np.mean(device_scores, axis=1)
                +
                0.5 * np.max(device_scores, axis=1)
            )

        elif aggregation == "topk":

            k = config["evaluation"].get(
                "topk_devices",
                3
            )

            topk = np.sort(
                device_scores,
                axis=1
            )[:, -k:]

            timestamp_scores = np.mean(
                topk,
                axis=1
            )
            
        else:
            raise ValueError(
                f"Unknown score aggregation: {aggregation}"
            )

        scores[split] = timestamp_scores

    # ----------------------------------------
    # Optional temporal smoothing
    # ----------------------------------------

    if config["evaluation"].get(
        "score_smoothing_enabled",
        False
    ):

        window = config["evaluation"].get(
            "score_smoothing_window",
            5
        )

        for split in scores.keys():

            s = scores[split]

            kernel = np.ones(window) / window

            scores[split] = np.convolve(
                s,
                kernel,
                mode="same"
            )

    np.savez(
        f"{eval_results_folder}/scores.npz",
        **scores
    )

    print("Scores saved.")


def smooth_scores(scores, window=5):
    """
    smooth anomaly scores.

    Parameters:
    - window: size of the smoothing window (in timestamps)
    """
    print("Smoothing scores...")

    # 🔹 smooth scores
    for split in scores:
        for score_type in scores[split]:
            scores[split][score_type] = np.convolve(scores[split][score_type], np.ones(window)/window, mode='same')

    return scores