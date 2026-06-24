import matplotlib.pyplot as plt
import os

def plot_scores(scores, exp_dir):

    plt.figure(figsize=(15,5))
    plt.plot(scores, label="Anomaly Scores")
    plt.legend()
    plt.title("Anomaly Scores")

    path = os.path.join(exp_dir, "anomaly_plot.png")
    plt.savefig(path)

    plt.close()