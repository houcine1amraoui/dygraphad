import yaml

from src.utils.create_train_val_loaders import create_train_val_loaders
from src.utils.seed import set_seed
from src.models.builder import build_model
from src.training.trainer import train
import argparse

def main_train():
    # 1. Set configuration
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

    # Create a folder for experiments (per dataset, per model, per time)
    # train_experiments_time_folder = create_train_experiments_folder(config)
    
    # 2. Dataset/DataLoader creation
    train_loader, val_loader = create_train_val_loaders(config)

    # 3. Model Initialization
    model = build_model(config)

    # 3. Start training
    train(model, train_loader, val_loader, config)
  
if __name__ == "__main__":
    main_train()