from main_preprocess import main_preprocess
from main_train import main_train
from main_evaluation import main_evaluation
from main_visualization import main_visualization

def main():
    main_preprocess()
    main_train()
    main_evaluation()
    main_visualization()

if __name__ == "__main__":
    main()