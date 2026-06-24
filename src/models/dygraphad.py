import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """
    Encode one graph A ∈ R^(N×N)
    into graph embedding h ∈ R^H
    """

    def __init__(self, num_nodes, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(num_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A):
        """
        A: (B,N,N)
        """

        h = F.relu(self.fc1(A))
        h = F.relu(self.fc2(h))

        # global graph pooling
        h = h.mean(dim=1)

        return h
        

class DyGraphAD(nn.Module):

    def __init__(
        self,
        num_nodes,
        window_size=30,
        graph_history=5,
        hidden_dim=128,
        transformer_heads=4,
        transformer_layers=2
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.window_size = window_size
        self.graph_history = graph_history
        self.hidden_dim = hidden_dim

        # ==================================================
        # Graph Encoder
        # ==================================================

        self.graph_encoder = GraphEncoder(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim
        )

        # ==================================================
        # Graph Transformer
        # ==================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        self.graph_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # ==================================================
        # Graph Decoder
        # ==================================================

        self.graph_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )

        # ==================================================
        # TS Encoder
        # ==================================================

        self.gru = nn.GRU(
            input_size=num_nodes,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ==================================================
        # Forecast Head
        # ==================================================

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, batch):

        """
        batch["x"]
            (B,W,N)

        batch["graphs"]
            (B,M,N,N)
        """

        x = batch["x"]
        graphs = batch["graphs"]

        B, M, N, _ = graphs.shape

        # ==================================================
        # Encode graph sequence
        # ==================================================

        graph_embeddings = []

        for t in range(M):

            g = graphs[:, t]

            emb = self.graph_encoder(g)

            graph_embeddings.append(emb)

        graph_embeddings = torch.stack(
            graph_embeddings,
            dim=1
        )

        # (B,M,H)

        # ==================================================
        # Graph Forecast Branch
        # ==================================================

        z = self.graph_transformer(
            graph_embeddings
        )

        graph_context = z[:, -1]

        pred_graph = self.graph_decoder(
            graph_context
        )

        pred_graph = pred_graph.view(
            B,
            N,
            N
        )

        # ==================================================
        # TS Forecast Branch
        # ==================================================

        _, h = self.gru(x)

        ts_context = h[-1]

        fusion = ts_context + graph_context

        pred = self.forecast_head(
            fusion
        )

        return {
            "pred": pred,
            "pred_graph": pred_graph
        }

    def loss(
        self,
        batch,
        output,
        lambda_graph=1.0
    ):

        y = batch["y"]

        graph_target = batch["graph_target"]

        pred = output["pred"]

        pred_graph = output["pred_graph"]

        ts_loss = F.mse_loss(
            pred,
            y
        )

        graph_loss = F.mse_loss(
            pred_graph,
            graph_target
        )

        return (
            ts_loss
            + lambda_graph * graph_loss
        )

    @torch.no_grad()
    def compute_anomaly_errors(
        self,
        batch,
        output=None
    ):

        if output is None:
            output = self(batch)

        pred = output["pred"]

        pred_graph = output["pred_graph"]

        y = batch["y"]

        graph_target = batch["graph_target"]

        # ---------------------------------
        # TS forecast error
        # ---------------------------------

        forecast_error = torch.abs(
            pred - y
        )

        # (B,N)

        # ---------------------------------
        # Graph error
        # ---------------------------------

        graph_error = torch.mean(
            torch.abs(
                pred_graph
                - graph_target
            ),
            dim=2
        )

        # (B,N)

        # ---------------------------------
        # Combined
        # ---------------------------------

        combined_error = (
            forecast_error
            + graph_error
        )

        # this my own modificaiton
        return forecast_error + graph_error + combined_error
    
        return {
            "forecast": forecast_error,
            "graph": graph_error,
            "combined": combined_error
        }