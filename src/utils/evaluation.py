import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Define a wrapper for agent invocation with Exponential Backoff and Jitter
# This helps handle TPM (Tokens Per Minute) and RPM (Requests Per Minute) limits.
@retry(
    wait=wait_exponential_jitter(initial=1, max=60, jitter=1),
    stop=stop_after_attempt(15),
    reraise=True
)
def invoke_agent_with_retry(agent, state):
    """
    Invokes the agent with retry logic using exponential backoff with jitter.
    """
    return agent.invoke(state)

def run_evaluation(agent, node_df, node_df_normal, ground_truth_df, market_df, window_size=30, step_size=1, limit=None):
    """
    Runs the teacher agent on the provided data in a sliding window fashion.
    Results are automatically saved by the agent's logger node.
    
    Includes Exponential Backoff with Jitter for agent invocation.
    """
    print("Starting evaluation run...")
    
    idx = pd.IndexSlice

    # 1. Prepare indices
    if 'time' in node_df.columns:
        times = sorted(node_df['time'].unique())
    elif isinstance(node_df.index, pd.MultiIndex):
        times = sorted(node_df.index.get_level_values('time').unique())
    else:
        # Fallback if time structure is different
        try:
             times = sorted(node_df.index.unique())
        except:
             raise ValueError("Could not determine time index from node_df")
    
    if 'time' in node_df_normal.columns:
        times = sorted(node_df_normal['time'].unique())
    elif isinstance(node_df_normal.index, pd.MultiIndex):
        times = sorted(node_df_normal.index.get_level_values('time').unique())
    else:
        # Fallback if time structure is different
        try:
             times = sorted(node_df_normal.index.unique())
        except:
             raise ValueError("Could not determine time index from node_df")
    
    # Ensure data is indexed correctly
    if 'node_id' in node_df.columns and 'time' in node_df.columns:
        node_df_indexed = node_df.set_index(['node_id', 'time']).sort_index()
    else:
        node_df_indexed = node_df # Assuming already indexed

    if 'node_id' in node_df_normal.columns and 'time' in node_df_normal.columns:
        node_df_indexed_normal = node_df_normal.set_index(['node_id', 'time']).sort_index()
    else:
        node_df_indexed_normal = node_df_normal # Assuming already indexed
        
    if 'time' in ground_truth_df.columns:
        ground_truth_indexed = ground_truth_df.set_index('time').sort_index()
    else:
        ground_truth_indexed = ground_truth_df
    
    # 2. Iterate
    count = 0
    # Calculate total iterations
    total_windows = (len(times) - window_size) // step_size
    
    state = {}
    state["n_nodes"] = 6
    state["win_len"] = window_size

    for t_idx in tqdm(range(0, len(times) - window_size, step_size), total=total_windows):
        if limit is not None and count >= limit:
            break
            
        t_start = times[t_idx]
        t_end = times[t_idx + window_size - 1] # Inclusive range for loc usually
        
        # Slice for this window (using time range is safer for MultiIndex)
        # Note: We need a continuous range of 30 steps
        # Construct date range for robustness
        current_times = times[t_idx : t_idx + window_size]
        
        # Observation (Node Data)
        obs_window = node_df_indexed.loc[idx[:, current_times], :]
        
        # Ground Truth
        gt_window = ground_truth_indexed.loc[current_times]
        
        obs_window_normal = node_df_indexed_normal.loc[idx[:, current_times], :]
        
        # Prepare State
        state["obs_attacked"] = obs_window
        state["obs_normal"] = obs_window_normal
        state["ground_truth"] = gt_window
        
        # Invoke Agent with Retry
        try:
            # We don't need the return value as the logger saves it, but we capture it for printing/debugging
            new_state = invoke_agent_with_retry(agent, state)
            
            # Optional: Print basic info to monitor progress
            print("New State -> true_label: ", new_state.get("true_label"))
            print("Detection Label: ", new_state.get("final_market_attack_status"))

            state["node_level_attention_prev"] = new_state["node_level_attention_prev"]
            state["entropy_scores_prev"] = new_state["entropy_scores_prev"]
            state["node_level_attention_prev_normal"] = new_state["node_level_attention_prev_normal"]
            state["entropy_scores_prev_normal"] = new_state["entropy_scores_prev_normal"]
            
        except Exception as e:
            print(f"Error at window {t_start} - {t_end} after retries: {e}")
            
        count += 1
        
    print(f"Evaluation finished. Processed {count} windows.")

def calculate_metrics(log_path="teacher_experience_new/data.parquet"):
    """
    Reads the logs and computes performance metrics.
    """
    if not os.path.exists(log_path):
        print("No log file found.")
        return None

    df = pd.read_parquet(log_path)
    
    # Map string labels to binary (Attack=1, Non-Attack=0)
    # Adjust mapping based on your exact label strings
    if "true_label" not in df.columns or "final_market_attack_status" not in df.columns:
         print("Required columns 'true_label' or 'final_market_attack_status' missing in logs.")
         return df

    y_true = df["true_label"].apply(lambda x: 1 if x == "attacked" else 0)
    y_true_np = np.array(y_true)
    
    # Prediction might be "attacked", "non-attacked", "suspicious"
    # Treat "suspicious" as positive (attack) or negative? Usually positive for safety.
    y_pred = df["final_market_attack_status"].apply(lambda x: 1 if x in ["attacked", "suspicious"] else 0)
    y_pred_np = np.array(y_pred)
    
    # Compute Metrics
    try:
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0, average="weighted")
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
        if (tp+fn) > 0:
            TPR = tp/(tp+fn)
        else:
            TPR = 0
        if (tp+fp) > 0:
            precision = tp/(tp+fp)
        else:
            precision = 0
        if (fp+tn) > 0:
            FPR = fp/(fp+tn)
        else:
            FPR = 0
        if (fn+tp) > 0:
            FNR = fn/(fn+tp)
        else:
            FNR = 0
        print("\n--- Performance Report ---")
        print(f"Total Samples: {len(df)}")
        print(f"F1 Score: {f1:.4f}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print("\nClassification Report:")
        print("TPR= ", TPR)
        print("precision= ", precision)
        print("FPR= ", FPR)
        print("FNR= ", FNR)
        # Ensure classification_report knows about both classes even if one is missing in the data
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Non-Attacked", "Attacked"], zero_division=0))
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    return df

