import torch
from torch.utils.data import Dataset
import numpy as np
from dtaidistance import dtw


def build_dtw_graph(segment):
    """
    segment: (w, N)

    returns:
        (N, N)
    """

    N = segment.shape[1]

    graph = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i, N):

            d = dtw.distance(
                segment[:, i],
                segment[:, j]
            )

            sim = np.exp(-d)

            graph[i, j] = sim
            graph[j, i] = sim

    return graph

def build_graph_sequence(window, m=5):
    """
    window: (30,N)

    returns:
        (m,N,N)
    """

    segment_length = window.shape[0] // m

    graphs = []

    for k in range(m):

        start = k * segment_length
        end = (k + 1) * segment_length

        segment = window[start:end]

        G = build_dtw_graph(segment)

        graphs.append(G)

    return np.stack(graphs)

class DyGraphADDataset(Dataset):

    def __init__(
        self,
        data,
        window_size=30,
        m=5
    ):
        self.data = data
        self.window_size = window_size
        self.m = m

    def __len__(self):
        return len(self.data) - self.window_size - 1

    def __getitem__(self, idx):

        x = self.data[
            idx:
            idx + self.window_size
        ]

        y = self.data[
            idx + self.window_size
        ]

        current_graphs = build_graph_sequence(
            x,
            self.m
        )

        future_window = self.data[
            idx + 1:
            idx + self.window_size + 1
        ]

        next_graphs = build_graph_sequence(
            future_window,
            self.m
        )

        return {
            "x": torch.tensor(
                x,
                dtype=torch.float32
            ),
            "y": torch.tensor(
                y,
                dtype=torch.float32
            ),
            "graphs": torch.tensor(
                current_graphs,
                dtype=torch.float32
            ),
            "graph_target": torch.tensor(
                next_graphs[-1],
                dtype=torch.float32
            )
        }