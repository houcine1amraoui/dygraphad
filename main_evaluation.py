import yaml
import argparse

from src.utils.seed import set_seed
from src.evaluation.compute_errors import compute_errors
from src.evaluation.compute_scores import compute_scores
from src.evaluation.compute_metrics import compute_point_wise_metrics, compute_segment_wise_metrics

def main_evaluation():
    # Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str)
    args = parser.parse_args()

    # override project_root_directory
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    # Evaluation pipeline: 
    # raw errors → normalize errors 
    # → scores (aggregated errors per timestamp) → smoothed scores
    # → metrics (point-wise)
    compute_errors(config)
    compute_scores(config)
    compute_point_wise_metrics(config)
    # compute_segment_wise_metrics(config)

if __name__ == "__main__":
    main_evaluation()

