from typing import TypedDict, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch

class FDIState(TypedDict, total=False):
    obs_normal: pd.DataFrame
    obs_attacked: pd.DataFrame
    ground_truth : pd.DataFrame

    n_nodes: int
    win_len: int

    true_label : Optional[str]
    target_nodes : Optional[List]
    attacker_nodes : Optional[List]

    gat_att: Optional[np.ndarray]
    gat_att_normal: Optional[np.ndarray]

    GAT_reconstruction_error_normal: Optional[np.ndarray]
    GAT_reconstruction_error: Optional[np.ndarray]

    node_level_attention_prev: Optional[dict]   
    entropy_scores_prev: Optional[dict]     

    node_level_attention_prev_normal: Optional[dict]   
    entropy_scores_prev_normal: Optional[dict]        

    gat_interpreter_output: Optional[str]
    obs_txt: Optional[str]
    
    use_tool: bool
    market_attack_status: Optional[str]
    attacked_nodes: Optional[List[int]]
    confidence : Optional[float]
    explanation: Optional[str]
    
    final_market_attack_status: Optional[str]
    final_attacked_nodes: Optional[List[int]]
    final_confidence : Optional[float]
    final_explanation: Optional[str]

    llm_input : Optional[dict]

@dataclass
class Config:
    # Time/grid settings
    freq: str = "15T"                 # sampling interval ("T"=1min, "15T"=15min)
    input_window: int = 30             
    horizon: int = 1                   # forecast t + horizon
    knn: int = 3                       # neighbors per node in similarity graph

    # Prices
    FiT: float = 0.991      # FiT ($/kWh) — OK to keep in config
    include_price_features: bool = True# if True, append [P_buy(t), P_sell] as features per node

    # Training
    epochs: int = 200
    batch_size: int = 1                # one time‑snapshot (all nodes) per batch
    lr: float = 5e-4
    hidden_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 1e-5
    patience: int = 10                 # Early stopping patience

    # Misc
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    results_dir: str = "results"
