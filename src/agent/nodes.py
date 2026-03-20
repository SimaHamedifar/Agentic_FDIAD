import pandas as pd
import numpy as np
import torch
import math
import copy
import joblib
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
import re

from src.agents.state import FDIState, Config
from src.utils.helpers import to_json_safe, parse_llm_json

# Placeholder for utils and models. Adjust these imports if they change.
# Assuming these are available in PYTHONPATH or we will provide them / they already exist.
# We will add a mock edge_index for now or import it if needed. 

# Let's see what edge_index was in practice.py
edge_index = torch.tensor([[4, 3, 5, 0, 2, 1, 2, 1, 4, 5, 0, 2, 1, 0, 2, 1, 3, 4, 3, 5, 4, 5, 2, 1, 2],
        [0, 1, 1, 2, 2, 0, 5, 3, 2, 3, 1, 4, 2, 4, 1, 5, 2, 1, 5, 2, 4, 5, 0, 4, 3]])

# Assuming the following are available in the system
try:
    from Utils.scale_test_data import TestScaler
    from gat_model import GATMarket
    from gat_encoder_wrapper import GATEncoderWrapper
    from Utils.p2pdataset_for_gat import P2PDataset
except ImportError:
    pass

def input_node(state: dict) -> FDIState:
    """
    Clean and correct input node.
    Takes an incoming state dict and normalizes:
    - obs → LangGraph Dataframe wrapper
    - n_nodes, win_len → passed through
    - other state fields initialized to None
    """
    obs_df = state["obs_attacked"]
    ground_truth_df = state["ground_truth"]

    last_label = ground_truth_df.iloc[-1]["label"]
    label = "attacked" if last_label == "attacked" else "non-attacked"

    attacker_nodes = []
    for item in ground_truth_df["attacker_nodes"]:
        if item not in attacker_nodes:
            attacker_nodes.append(item)
    
    target_nodes = []
    for item in ground_truth_df["target_nodes"]:
        if item not in target_nodes:
            target_nodes.append(item)

    runs = 0
    round = 0

    return {**state,
        "true_label": label,
        "target_nodes": target_nodes,
        "attacker_nodes": attacker_nodes,
        "tool_use": round,
        "llm_runs"  : runs,
        "safety_checker_output": None,
        "first_stage": False,
        "second_stage": False,
        "final_stage": False
    }

class GATNode ():
    def __init__(self, file_path):
        self.feature_cols = ['time_sin', 'time_cos', 'demand', 'generation', 'net_load', 'y_settled', 'in_flow', 'out_flow']
        self.target_cols = ['demand', 'generation', 'net_load']
        self.test_scaler = TestScaler(feature_columns=self.feature_cols)
        self.file_path = file_path

    def __call__(self, state:FDIState) -> FDIState:
        
        df = state["obs_attacked"]
        df_normal = state["obs_normal"]
        scaled_df = self.test_scaler.scale_test_data(df)
        scaled_df_normal = self.test_scaler.scale_test_data(df_normal)

        idx = pd.IndexSlice
        Xt = {}
        Xt_normal = {}
        yt = {}
        yt_normal = {}
        X_list = []
        X_list_normal = []
        nodes = scaled_df.index.get_level_values('node_id').unique()
        times = scaled_df.index.get_level_values('time').unique()
        t_end = times[-1]
        for node in nodes:
            df_hist = scaled_df.loc[idx[node, :], self.feature_cols]
            df_hist_normal = scaled_df_normal.loc[idx[node, :], self.feature_cols]
            x = torch.tensor(df_hist.values, dtype=torch.float32)  # (win, F)
            x_normal = torch.tensor(df_hist_normal.values, dtype=torch.float32)  # (win, F)
            X_list.append(x.reshape(-1))  # flatten into (win*F)
            X_list_normal.append(x_normal.reshape(-1))  # flatten into (win*F)
        X = torch.stack(X_list, dim=0)  # shape (N, win*F)
        X_normal = torch.stack(X_list_normal, dim=0)  # shape (N, win*F)
        target_df = scaled_df.loc[idx[slice(None), t_end], self.target_cols]
        target_df_normal = scaled_df_normal.loc[idx[slice(None), t_end], self.target_cols]
        y = torch.tensor(target_df.values, dtype=torch.float32)  # (N, 3)
        y_normal = torch.tensor(target_df_normal.values, dtype=torch.float32)  # (N, 3)
        Xt[t_end] = X   # input features
        Xt_normal[t_end] = X_normal   # input features
        yt[t_end] = y   # targets
        yt_normal[t_end] = y_normal   # targets

        valid_times_train = list(Xt.keys())
        valid_times_train_normal = list(Xt_normal.keys())
        data = P2PDataset(Xt, yt, valid_times_train, edge_index)
        data_normal = P2PDataset(Xt_normal, yt_normal, valid_times_train_normal, edge_index)
        input_dim = Config.input_window * (len(self.feature_cols))
        model = GATMarket(in_dim=input_dim, cfg=Config)
        device = torch.device("cpu")
        checkpoint = torch.load(self.file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        emb_att_wrapper = GATEncoderWrapper(gat_model=model)
        embeddings_list, att_list, all_recons, all_inputs = emb_att_wrapper.encode_window(GeoDataset=data)
        model.eval()
        emb_att_wrapper = GATEncoderWrapper(gat_model=model)
        embeddings_list_normal, att_list_normal, all_recons_normal, all_inputs_normal = emb_att_wrapper.encode_window(GeoDataset=data_normal)
        TN, Fw = all_recons.shape  
        N = 6
        T = TN // N
        w = Config.input_window
        F = Fw // w
        
        x_hat_reshaped = all_recons.reshape(T, N, w, F)
        x_true_reshaped = all_inputs.reshape(T, N, w, F)

        x_hat_reshaped_normal = all_recons_normal.reshape(T, N, w, F)
        x_true_reshaped_normal = all_inputs_normal.reshape(T, N, w, F)

        x_hat_inv = np.zeros_like(x_hat_reshaped)
        x_true_inv = np.zeros_like(x_true_reshaped)

        x_hat_inv_normal = np.zeros_like(x_hat_reshaped_normal)
        x_true_inv_normal = np.zeros_like(x_true_reshaped_normal)

        filename= "Utils/scalers_6_Nodes.joblib"
        scalers = joblib.load(filename)
        for i, node in enumerate(nodes):
            scaler = scalers[node]
            for t in range(w):
                x_hat_inv[:, i, t, :] = scaler.inverse_transform(x_hat_reshaped[:, i, t, :])
                x_true_inv[:, i, t, :] = scaler.inverse_transform(x_true_reshaped[:, i, t, :])    

                x_hat_inv_normal[:, i, t, :] = scaler.inverse_transform(x_hat_reshaped_normal[:, i, t, :])
                x_true_inv_normal[:, i, t, :] = scaler.inverse_transform(x_true_reshaped_normal[:, i, t, :])    

        errors_n = np.mean((x_true_inv - x_hat_inv)**2, axis=(2,3))
        errors_n_normal = np.mean((x_true_inv_normal - x_hat_inv_normal)**2, axis=(2,3))

        normalized_errors_n = errors_n / np.sum(errors_n) if np.sum(errors_n) > 0 else errors_n
        normalized_errors_n_normal = errors_n_normal / np.sum(errors_n_normal) if np.sum(errors_n_normal) > 0 else errors_n_normal

        return {**state,"gat_att": att_list, "gat_att_normal": att_list_normal,
                 "GAT_reconstruction_error": normalized_errors_n, "GAT_reconstruction_error_normal": normalized_errors_n_normal}


class gat_interpreter():
    def __init__(self):
        pass

    # -----------------------------------------------------------
    # Node Attention Map
    # -----------------------------------------------------------
    def build_node_attention_map(self, att_list): # att_weights: the attention weights per time window
        edge_level_attention = {} # aggregated over heads
        node_level_attention = {}
        for pair in att_list:
            edge_index = pair[0]
            alpha = pair[1]
            src, dst = edge_index
            src = src.numpy()
            dst = dst.numpy()
            alpha = alpha.detach().numpy()
            alpha = alpha.mean(axis=1)
            for k in range(len(src)):
                i = int(src[k])
                j = int(dst[k])
                edge = (i, j)
                val = float(alpha[k])
                if edge not in edge_level_attention:
                    edge_level_attention[edge] = []
                edge_level_attention[edge].append(val)
        aggregated_edge_level = {key: float(sum(val)/len(val)) for key, val in edge_level_attention.items()}

        for key, val in aggregated_edge_level.items():
            i = key[0]
            j = key[1]
            if i not in node_level_attention:
                node_level_attention [i] = []
            node_level_attention[i].append((j,val))
        return aggregated_edge_level, node_level_attention
    
    # -----------------------------------------------------------
    # Attention Entropy
    # -----------------------------------------------------------
    def attention_entropy(self):
        """
        Compute entropy H = -sum α_ij log α_ij for each node.
        """
        entropy_scores = {}

        for node, neighs in self.node_level_attention.items():
            alphas = [a for _, a in neighs]
            H = 0.0
            for a in alphas:
                if a > 1e-9:
                    H += -a * math.log(a)
            entropy_scores[node] = H

        return entropy_scores
    
    # -----------------------------------------------------------
    # Attention drift: |α(t) - α(t-1)| summed over all neighbors
    # -----------------------------------------------------------
    def attention_drift(self, node_att_t_prev):
        """
        Compute drift between attention vectors across two time steps.
        """
        drift_scores = {}

        for node in self.node_level_attention:
            # Convert to dict for matching
            at = dict(self.node_level_attention[node])
            ap = dict(node_att_t_prev[node])

            # Union of neighbors
            neighbors = set(at.keys()) | set(ap.keys())

            drift = 0.0
            for j in neighbors:
                drift += abs(at.get(j, 0.0) - ap.get(j, 0.0))

            drift_scores[node] = drift

        return drift_scores
    
    # -----------------------------------------------------------
    # KL divergence between attention distributions
    # -----------------------------------------------------------
    def attention_kl(self, node_att_t_prev):
        kl_scores = {}

        for node in self.node_level_attention:
            at = dict(self.node_level_attention[node])
            ap = dict(node_att_t_prev[node])

            neighbors = set(at.keys()) | set(ap.keys())

            # normalize both distributions
            sum_t = sum(at.get(j, 0.0) for j in neighbors) + 1e-9
            sum_p = sum(ap.get(j, 0.0) for j in neighbors) + 1e-9

            kl = 0.0
            for j in neighbors:
                p = ap.get(j, 0.0) / sum_p + 1e-9
                q = at.get(j, 0.0) / sum_t + 1e-9
                kl += p * math.log(p / q)

            kl_scores[node] = kl

        return kl_scores
    
    # -----------------------------------------------------------
    # Final fusion score for FDI detection
    # -----------------------------------------------------------
    def compute_fdi_score(self, drift_scores, entropy_scores, entropy_prev, 
                        kl_scores, w1=0.4, w2=0.3, w3=0.3):
        """
        Simple anomaly fusion:
        S = w1 * Drift + w2 * |ΔEntropy| + w3 * KL + w4 * EmbDrift
        """
        fdi_score = {}

        for node in drift_scores:
            d = drift_scores[node]
            de = abs(entropy_scores[node] - entropy_prev[node])
            k = kl_scores[node]

            score = w1 * d + w2 * de + w3 * k
            fdi_score[node] = score

        total_sum = sum(fdi_score.values())
        if total_sum > 0:
            normalized_scores = {node_id: val / total_sum for node_id, val in fdi_score.items()}
        else:
            normalized_scores = fdi_score
        return fdi_score
        
    def embeddings_drift(self, embeddings_t, embeddings_t_prev):
        if embeddings_t_prev is None:
            return np.zeros(embeddings_t.shape[0])
        return np.linalg.norm(embeddings_t - embeddings_t_prev, axis=1)

    def reconstruction_error(self, ):
        return 

    # -----------------------------------------------------------
    # Call
    # -----------------------------------------------------------
    def __call__(self, state):
        att_list = state.get("gat_att") or []
        att_list_normal = state.get("gat_att_normal") or []
        
        self.att = att_list[0] if len(att_list) > 0 else None
        self.att_normal = att_list_normal[0] if len(att_list_normal) > 0 else None

        edge_att , self.node_level_attention = self.build_node_attention_map(self.att)
        edge_att_normal , self.node_level_attention_normal = self.build_node_attention_map(self.att_normal)
        entropy_scores = self.attention_entropy()
        entropy_scores_normal = self.attention_entropy()

        node_att_prev = state.get("node_level_attention_prev")
        node_att_prev_normal = state.get("node_level_attention_prev_normal")
        entropy_scores_prev = state.get("entropy_scores_prev")
        entropy_scores_prev_normal = state.get("entropy_scores_prev_normal")

        if node_att_prev is None:
            drift_scores = {n: 0.0 for n in self.node_level_attention}
            kl_scores = {n: 0.0 for n in self.node_level_attention}
            entropy_scores_prev = {n: entropy_scores[n] for n in self.node_level_attention}
        else:
            drift_scores = self.attention_drift(node_att_prev)
            kl_scores = self.attention_kl(node_att_prev)

        if node_att_prev_normal is None:
            drift_scores_normal = {n: 0.0 for n in self.node_level_attention_normal}
            kl_scores_normal = {n: 0.0 for n in self.node_level_attention_normal}
            entropy_scores_prev_normal = {n: entropy_scores_normal[n] for n in self.node_level_attention_normal}
        else:
            drift_scores_normal = self.attention_drift(node_att_prev_normal)
            kl_scores_normal = self.attention_kl(node_att_prev_normal)

        fdi_score = self.compute_fdi_score(drift_scores=drift_scores, 
                                        entropy_scores=entropy_scores, 
                                        entropy_prev=entropy_scores_prev,
                                        kl_scores=kl_scores)
        
        fdi_score_normal = self.compute_fdi_score(drift_scores=drift_scores_normal, 
                                        entropy_scores=entropy_scores_normal, 
                                        entropy_prev=entropy_scores_prev_normal,
                                        kl_scores=kl_scores_normal)
        
        reconstruction_error = state.get("GAT_reconstruction_error")
        reconstruction_error_normal = state.get("GAT_reconstruction_error_normal")
        gat_interpreter_text = (
            f"The connected prosumers in the trading market are listed in pairs as follows."
            f"{list(edge_att.keys())}"
            f"For this market of 6 nodes a graph attention network is trained using non-attacked data."
            f"Its attention weights between two consecutive time windows of input data and input data reconstruction error for both attacked and non attacked conditions are analyzed in the following metrics."
            f"Considering both conditions will give you more insight of the presence of any attacks."
            f"Assuming that the market is under no attack the metrics are as follows."
            f"The attantion weights drift is {drift_scores_normal}."
            f"The attention entropy change is {entropy_scores_normal}."
            f"The KL divergence between attention distributions is {kl_scores_normal}."
            f"The graph's reconstruction error is {reconstruction_error_normal}."
            f"Assuming that the market is under attack the metrics are as follows."
            f"The attantion weights drift is {drift_scores}."
            f"The attention entropy change is {entropy_scores}."
            f"The KL divergence between attention distributions is {kl_scores}."
            f"The graph's reconstruction error is {reconstruction_error}."
        )

        return {**state,
                "node_level_attention_prev": copy.deepcopy(self.node_level_attention), 
                "entropy_scores_prev": entropy_scores.copy(), 
                "gat_interpreter_output": gat_interpreter_text,
                "node_level_attention_prev_normal": copy.deepcopy(self.node_level_attention_normal), 
                "entropy_scores_prev_normal": entropy_scores_normal.copy(), 
                }

def observation_translator(state: FDIState):
    obs = state["obs_attacked"]
    df = obs[['net_load','in_flow','out_flow']]
    obs_md = df.to_markdown()
    n_nodes = state["n_nodes"]
    win_len = state["win_len"]
    start = obs.index.get_level_values('time')[0]
    end = obs.index.get_level_values('time')[-1]

    primary_txt = (f"The analyzed data is a 15-minutely sampled data" 
                   f"from {start} to {end} about the state of the peer-to-peer energy trading."
                   f"So, your detection will be considered for the time step {end}."
                   f"Please, consider that the attacks are mostly happening from 8 a.m. to 11 a.m. and from 12 p.m. to 4 p.m. and the aother times there are no attacks. "
                   f"Node 0 is the attacker, but you find the attacked nodes based on the connections and the metrics."
                    )

    idx = pd.IndexSlice
    obs_n_ = "\n Observations:"
    for i in range(n_nodes):
        obs_n = obs.loc[idx[i,:],:]
        num_buying = len(np.where(obs_n.net_load > 0)[0])
        num_selling = len(np.where(obs_n.net_load < 0)[0])
        if num_buying == win_len:
            obs_n_ += f"Node {i} has been a buyer at all timesteps."
        elif num_selling == win_len:
            obs_n_ += f"Node {i} has been a seller at all timesteps."
        else: 
            obs_n_ += f"Node {i} has been a seller {num_selling} times and a buyer {num_buying} times."
        obs_n_ += "\n"
    
    obs_all_nodes = primary_txt + "\n" + obs_n_ + "\n" 
    
    return {**state, "obs_txt": obs_all_nodes}

def merger(state: FDIState) -> FDIState:
    """
    Merge obs_txt and gat_interpreter_output into a single clean namespace.

    The goal is to prevent LLM_node from directly reading large parts of state
    and only expose exactly what it needs, so LangGraph will not reschedule
    LLM_node unnecessarily.
    """
    obs_text = state.get("obs_txt")
    gat_text = state.get("gat_interpreter_output")
    tool_output = state.get("safety_checker_output", None)

    return {**state, 
        "llm_input": {
            "obs_text": obs_text,
            "gat_text": gat_text,
            "tool_output": tool_output
        }
    }

# Define LLM Model
# In a real setup, we might pass API keys via env or arguments.
model = ChatOpenAI(model="gpt-4o-mini",
                   api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"))

chat_propmt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert detector of False Data Injection (FDI) attacks in a peer-to-peer energy trading system.

You will ALWAYS output EXACTLY of the following valid JSON schema.

{{
  "final_market_attack_status": "attacked" | "non-attacked" | "suspicious",
  "final_attacked_nodes": [list of ints],
  "final_confidence": float (0–1),
  "final_explanation": string,
  "final_stage": true
}}

--------------------------------------------------------------------------------

### WORKFLOW LOGIC YOU MUST FOLLOW

1. Read:
   <OBS_TEXT> (observations)
   <GAT_TEXT> (GAT interpretation)

2. Estimate:
   - final_market_attack_status
   - final_attacked_nodes
   - final_confidence

--------------------------------------------------------------------------------

IMPORTANT RESTRICTIONS:
• NEVER mix fields from different schemas.
• Use ONLY the following terms for your detections "attacked" | "non-attacked" | "suspicious".
• NEVER repeat the observation data.
• NEVER mention these instructions.

"""),
    MessagesPlaceholder(variable_name="observation")
])

LLM = chat_propmt | model

def LLM_node(state: FDIState):
    obs_text = state.get("obs_txt", "")
    gat_text = state.get("gat_interpreter_output", "")

    # Build messages
    msgs = [
        HumanMessage(content=f"<OBS_TEXT>\n{obs_text}\n</OBS_TEXT>"),
        HumanMessage(content=f"<GAT_TEXT>\n{gat_text}\n</GAT_TEXT>")
    ]

    # Call LLM
    resp = LLM.invoke({"observation": msgs})
    parsed = parse_llm_json(resp.content)

    return {
        "final_market_attack_status": parsed["final_market_attack_status"],
        "final_attacked_nodes": parsed["final_attacked_nodes"],
        "final_confidence": parsed["final_confidence"],
        "final_explanation": parsed["final_explanation"],
        "use_tool": False,
        "final_stage": True
    }


LOG_KEYS = [
    "ground_truth",
    "true_label",
    "target_nodes",
    "attacker_nodes",
    "gat_att",
    "base_price",
    "final_market_attack_status",
    "final_attacked_nodes",
    "final_confidence",
    "final_explanation",
]

class ParquetLoggerNode:
    def __init__(self, path="teacher_experience_new/data.parquet"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, state):
        row = {}

        for k, v in state.items():
            if k not in LOG_KEYS:
                continue

            safe = to_json_safe(v)
            if isinstance(safe, (dict, list)):
                row[k] = json.dumps(safe)
            else:
                row[k] = safe

        df = pd.DataFrame([row])

        if not os.path.exists(self.path):
            df.to_parquet(self.path, index=False)
        else:
            existing = pd.read_parquet(self.path)
            pd.concat([existing, df], ignore_index=True).to_parquet(self.path, index=False)

        return {}
