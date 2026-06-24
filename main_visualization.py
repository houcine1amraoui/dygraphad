import yaml

from src.utils.seed import set_seed
from src.visualization.viz import plot_boxplot, plot_anomaly_scores_distribution, plot_bins

def main_visualization():
    # 1. Set configuration
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])
    
    plot_anomaly_scores_distribution(config)
    plot_boxplot(config)
    plot_bins(config)

if __name__ == "__main__":
    main_visualization()