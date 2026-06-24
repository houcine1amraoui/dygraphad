import json


from src.models.lstm_ae import LSTMAE
from src.models.lstm_vae import LSTMVAE
from src.models.usad import USAD
from src.models.omni_anomaly import OmniAnomaly
from src.models.dagmm_1 import DAGMM1
from src.models.dagmm_2 import DAGMM2

from src.models.fusagnet1 import FuSAGNet1
from src.models.gdn import GDN
from src.models.mtad_gat import MTAD_GAT

from src.utils.device import get_device 
from src.utils.get_folders_utils import get_processed_folder

def build_model(config):

    processed_data_folder = get_processed_folder(config)

    model_name = config["training"]["model"]

    with open(f"{processed_data_folder}/devices.json") as f:
        devices = json.load(f)

    device = get_device()
    model = None

    if model_name == "lstm_ae":
        model = LSTMAE(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            hidden_dim=128,
            latent_dim=64,
            num_layers=1
        ).to(device)

    elif model_name == "lstm_vae":
        model = LSTMVAE(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            hidden_dim=128,
            latent_dim=32,
            beta=0.001
        ).to(device)

    elif model_name == "usad":
        model = USAD(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            latent_dim=128
        ).to(device)

    elif model_name == "omni_anomaly":
        model = OmniAnomaly(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            hidden_dim=128,
            latent_dim=32,
            beta=0.001
        ).to(device)
        
    elif model_name == "dagmm1":
        model = DAGMM1(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            latent_dim=16,
            n_gmm=4
        ).to(device)

    elif model_name == "dagmm2":
        model = DAGMM2(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            latent_dim=16,
            n_gmm=4,
            lambda_energy=0.1,
            lambda_cov=0.005
        ).to(device)

    elif model_name == "fusagnet1":
        model = FuSAGNet1(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            latent_size=config["training"]["window_size"],
            hidden_dim=64,
            topk=15
        ).to(device)

    elif model_name == "gdn":
        params = config["models"][0]["params"]
        model = GDN(
            number_nodes=len(devices),
            in_dim=config["training"]["window_size"],
            hid_dim=params["hidden_dim"],
            topk=params["topk"],
            heads=params["heads"]
        ).to(device)
    elif model_name == "mtad_gat":
        params = config["models"][0]["params"]

        model = MTAD_GAT(
            n_features=len(devices),
            window_size=config["training"]["window_size"],
            out_dim=len(devices),

            kernel_size=params.get("kernel_size", 7),
            feat_gat_embed_dim=params.get("feat_gat_embed_dim", None),
            time_gat_embed_dim=params.get("time_gat_embed_dim", None),

            gru_n_layers=params.get("gru_n_layers", 1),
            gru_hid_dim=params.get("gru_hid_dim", 150),

            forecast_n_layers=params.get("forecast_n_layers", 1),
            forecast_hid_dim=params.get("forecast_hid_dim", 150),

            recon_n_layers=params.get("recon_n_layers", 1),
            recon_hid_dim=params.get("recon_hid_dim", 150),

            dropout=params.get("dropout", 0.2),
            alpha=params.get("alpha", 0.2)
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model