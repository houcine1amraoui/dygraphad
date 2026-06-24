from src.preprocessing.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import torch

from src.utils.device import get_device
from src.utils.experiment import load_best_checkpoint
from src.utils.get_folders_utils import get_processed_folder, get_evaluation_results_main_folder
from src.utils.create_folders_utils import create_eval_results_folder

def create_evaluation_dataloaders(config):
    processed_data_folder = get_processed_folder(config)
    
    # load config
    window_size = config["training"]["window_size"]
    batch_size = config["evaluation"]["batch_size"]

    arrays = np.load(f"{processed_data_folder}/arrays.npz")

    train_dataset = TimeSeriesDataset(arrays["train"], window_size)
    val_dataset = TimeSeriesDataset(arrays["val"], window_size)
    actor2_test_dataset = TimeSeriesDataset(arrays["actor2_test"], window_size)
    actor1_test_dataset = TimeSeriesDataset(arrays["actor1_test"], window_size)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    actor2_test_loader = DataLoader(actor2_test_dataset, batch_size, shuffle=True)
    actor1_test_loader = DataLoader(actor1_test_dataset, batch_size, shuffle=True)

    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "actor2_test_loader": actor2_test_loader,
        "actor1_test_loader": actor1_test_loader
    }
    return data_loaders

def compute_errors_per_loader(model, dataloader):
    """
    Compute anomaly errors for one dataloader.

    Returns:
        errors:
            numpy array of shape (T, N)
    """

    device = get_device()

    model.eval()

    all_errors = []

    with torch.no_grad():

        for x, y in tqdm(dataloader):

            x = x.to(device)
            y = y.to(device)

            batch = {
                "x": x,
                "y": y
            }
            
            errors = model.compute_anomaly_errors(
                batch=batch
            )

            all_errors.append(
                errors.cpu().numpy()
            )


    errors = np.concatenate(
        all_errors,
        axis=0
    )

    return errors

def compute_raw_errors_all_splits(config):

    """
    Compute anomaly errors for all splits and save them.
    """

    print("Computing anomaly errors for all splits...")

    model = load_best_checkpoint(config)

    data_loaders = create_evaluation_dataloaders(config)

    train_errors = compute_errors_per_loader(
        model,
        data_loaders["train_loader"]
    )

    val_errors = compute_errors_per_loader(
        model,
        data_loaders["val_loader"]
    )

    actor2_test_errors = compute_errors_per_loader(
        model,
        data_loaders["actor2_test_loader"]
    )

    actor1_test_errors = compute_errors_per_loader(
        model,
        data_loaders["actor1_test_loader"]
    )

    raw_errors = {
        "train": train_errors,
        "val": val_errors,
        "actor2_test": actor2_test_errors,
        "actor1_test": actor1_test_errors,
    }

    eval_results_folder = create_eval_results_folder(config)

    np.savez(f"{eval_results_folder}/raw_errors.npz",**raw_errors)

    print("Raw anomaly errors saved.")

    
def normalize_raw_errors_all_splits(config):
    """
    Normalize anomaly errors using train statistics only.

    Uses robust scaling:

        score = |(x - median) / IQR|

    Statistics are computed ONLY from train scores.
    """

    print("Normalizing raw errors for all splits...")

    eval_results_folder = get_evaluation_results_main_folder(config)

    data = np.load(
        f"{eval_results_folder}/raw_errors.npz"
    )

    train_scores = data["train"]

    # --------------------------------------------------
    # Robust statistics from TRAIN only
    # --------------------------------------------------

    median = np.median(
        train_scores,
        axis=0
    )

    iqr = (
        np.percentile(train_scores, 75, axis=0)
        -
        np.percentile(train_scores, 25, axis=0)
    )

    # avoid division by zero
    iqr = np.maximum(
        iqr,
        1.0
    )

    def normalize(scores):

        scores = (
            scores - median
        ) / iqr

        scores = np.clip(
            scores,
            -5,
            5
        )

        return np.abs(scores)

    normalized = {}

    for split in [
        "train",
        "val",
        "actor2_test",
        "actor1_test"
    ]:

        normalized[split] = normalize(
            data[split]
        )

    np.savez(
        f"{eval_results_folder}/norm_errors.npz",
        **normalized
    )

    print("Normalized errors saved.")

def compute_errors(config):
    # Compute errors for all loaders
    compute_raw_errors_all_splits(config)

    # normalize computed raw errors
    normalize_raw_errors_all_splits(config)