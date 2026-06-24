import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.dygraphad_graph_builder import build_graph_sequence
from src.utils.load_actors_timelines import load_actor_timelines
from src.utils.get_folders_utils import get_dataset_path

def clean_cu_data(df):
    """
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " with all NaN
    By default, df.dropna() removes any row that has at least one NaN. (we dont want that)
    """
    print("Cleaning CU data...")
    
    # Replace fake values with NaN
    # e.g., sensor.aqara_wireless_switch_pid_081_action → mean = -9.68
    # df.replace([-9.21, -9.68, -9.7, -9.81], pd.NA, inplace=True)
    
    # separate timestamp
    # timestamp_col = None

    # if "Timestamp" in df.columns:
    #     timestamp_col = df["Timestamp"]
    #     df = df.drop(columns=["Timestamp"])

    # keep only numeric columns
    # df = df.select_dtypes(include=['number'])

    # remove near-constant sensors
    # e.g., switch.kasa_023 → std = 0, this brings ZERO information
    # df = df.loc[:, df.std() > 1e-6]

    # restore timestamp
    # if timestamp_col is not None:
    #     df.insert(0, "Timestamp", timestamp_col)

    df = df.dropna(axis=1, how="all")   # remove dead sensors
    # df = df.fillna(method="ffill")      # or interpolate
    return df

def split_actor_periods(df, config):
    """
    Split dataset into train/val/test according to actor timelines.
    - actor1_train (normal training from Actor 1 timeline 1 only)
    - actor1_val (normal validation from Actor 1 timeline 1 only)
    - actor2_test (test from Actor 2 timeline)
    - act
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Load timelines and val_ratio from config
    timelines = load_actor_timelines(config)
    val_ratio = config["preprocessing"]["val_ratio"]

    # Slice data according to timelines
    actor1_t1 = df[
        (df["Timestamp"] >= timelines["actor1_t1"][0]) &
        (df["Timestamp"] <= timelines["actor1_t1"][1])
    ]

    actor2 = df[
        (df["Timestamp"] >= timelines["actor2"][0]) &
        (df["Timestamp"] <= timelines["actor2"][1])
    ]

    actor1_t2 = df[
        (df["Timestamp"] >= timelines["actor1_t2"][0]) &
        (df["Timestamp"] <= timelines["actor1_t2"][1])
    ]

    # Sort
    actor1_t1 = actor1_t1.sort_values("Timestamp")
    actor2_test_df = actor2.sort_values("Timestamp")
    actor1_test_df = actor1_t2.sort_values("Timestamp")

    # Split train/val
    split_idx = int(len(actor1_t1) * (1 - val_ratio))
    train_df = actor1_t1.iloc[:split_idx]
    val_df   = actor1_t1.iloc[split_idx:]

    splits = { "train": train_df, 
               "val": val_df, 
               "actor2_test": actor2_test_df, 
               "actor1_test": actor1_test_df
            }
    
    return splits

def normalize(splits, devices):
    """
    Normalize data using ONLY train_df (Actor1 timeline1)
    Returns:
    - train/val/test arrays (features only)
    - scaler
    """
    scaler = MinMaxScaler()

    # Fit scaler only on training data (features only without timestamp)
    # .to_numpy() is safer than .values() which removes column structure
    train_array_norm = scaler.fit_transform(splits["train"][devices].to_numpy())
    val_array_norm   = scaler.transform(splits["val"][devices].to_numpy())
    actor2_test_array_norm = scaler.transform(splits["actor2_test"][devices].to_numpy())
    actor1_test_array_norm = scaler.transform(splits["actor1_test"][devices].to_numpy())

    splits_norm = {
        "train":train_array_norm,
        "val": val_array_norm,
        "actor2_test":actor2_test_array_norm,
        "actor1_test": actor1_test_array_norm
    }
    return splits_norm, scaler

import numpy as np

def precompute_dygraphad_graphs(
    data,
    save_path,
    window_size=30,
    graph_history=5
):

    graph_sequences = []
    graph_targets = []

    T = len(data)

    for idx in range(
        T - window_size - 1
    ):

        current_window = data[
            idx:
            idx + window_size
        ]

        future_window = data[
            idx + 1:
            idx + window_size + 1
        ]

        current_graphs = build_graph_sequence(
            current_window,
            graph_history
        )

        future_graphs = build_graph_sequence(
            future_window,
            graph_history
        )

        graph_sequences.append(
            current_graphs
        )

        graph_targets.append(
            future_graphs[-1]
        )

    np.savez(
        save_path,
        graphs=np.asarray(graph_sequences),
        graph_targets=np.asarray(graph_targets)
    )

    print(
        f"Saved {len(graph_sequences)} graph samples"
    )

def load_data(config):
    data_path = get_dataset_path(config)
    dataset_name = config["preprocessing"]["dataset_name"]
    
    print(f"Loading {dataset_name} data...")
    df = pd.read_csv(data_path)
    return df

def preprocessing_pipeline(config):
    """
    Full preprocessing pipeline for GDN:
    1. Load CSV data
    2. Clean data
    3. Get device columns (exclude Timestamp)
    4. Split actors timelines (train/val/test)
    5. Normalize/Scale features (devices)
    6. Compute dynamic graphs for each sliding window (m graphs per window)
    7. Save processed data to disk

    Returns:
    - arrays.npz: contains train/val/actor2_test/actor1_test splits (features only)
    - timestamps.npz: timestamps arrays for each split (for plotting/reference)
    - scaler
    - devices.json: contains devices list
    """
   
    dataset_name = config["preprocessing"]["dataset_name"]

    # 1. Load CSV data
    df = load_data(config)
    
    # 2. Clean CU data 
    if dataset_name == "CU": df = clean_cu_data(df)
    
    # 3. Get device columns (exclude Timestamp)
    devices = [c for c in df.columns if c != "Timestamp"]
    print("nbr of devices:", len(devices))

    # 4. Split actors / train-val-test
    splits = split_actor_periods(df, config)
    print("Actor split done.")

    # 5. Save timestamps for reference/plotting
    train_timestamps = splits["train"]['Timestamp'].to_numpy()
    val_timestamps   = splits["val"]['Timestamp'].to_numpy()
    actor2_test_timestamps = splits["actor2_test"]['Timestamp'].to_numpy()
    actor1_test_timestamps = splits["actor1_test"]['Timestamp'].to_numpy()

    timestamps = {
        "train": train_timestamps,
        "val": val_timestamps,
        "actor2_test": actor2_test_timestamps,
        "actor1_test": actor1_test_timestamps
    }
    
    # 7. Normalize features
    splits_norm, scaler = normalize(splits, devices)
    print("Normalization done.")

    print("Train split:", len(splits_norm["train"]), f'% {len(splits_norm["train"])/len(df)*100:.2f}') 
    print("Validation split:",len(splits_norm["val"]), f'% {len(splits_norm["val"])/len(df)*100:.2f}')
    print("Actor 2 test split:",len(splits_norm["actor2_test"]), f'% {len(splits_norm["actor2_test"])/len(df)*100:.2f}')
    print("Actor 1 test split:",len(splits_norm["actor1_test"]), f'% {len(splits_norm["actor1_test"])/len(df)*100:.2f}')

    precompute_dygraphad_graphs(
        splits_norm["train"],
        "data/CU/train_graphs.npz",
        window_size=30,
        graph_history=5
    )
    return (splits_norm, timestamps, scaler, devices)