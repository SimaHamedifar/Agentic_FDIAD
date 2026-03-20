import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.agents.workflow import build_teacher_agent
from src.utils.evaluation import run_evaluation, calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Run FDI Detection Teacher Agent")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_gatmarket_SixNode_model.pth",
                        help="Path to GAT model checkpoint")
    parser.add_argument("--log_path", type=str, default="teacher_experience_new/data.parquet",
                        help="Path to save evaluation logs")
    parser.add_argument("--data_dir", type=str, default="Utils/Data",
                        help="Base directory for data files")
    
    args = parser.parse_args()

    print("Building teacher agent...")
    teacher_agent = build_teacher_agent(gat_model_path=args.model_path, log_path=args.log_path)

    # Note: These paths reflect the local setup used in the notebook.
    # In a real pipeline, we'd adjust them based on `--data_dir` or pass them explicitly.
    try:
        node_df_attacked_easy = pd.read_csv("node_df_0.3.csv")
        ground_truth_df_easy = pd.read_csv(f"{args.data_dir}/0.5/ground_truth_0.5_0_2.csv")
        node_df_normal = pd.read_csv(f"{args.data_dir}/obs_normal.csv")
        market_df_easy = pd.read_csv(f"{args.data_dir}/0.3/market_df_0.3_0_2.csv")
    except FileNotFoundError as e:
        print(f"Data files not found. Expected local files or paths relative to {args.data_dir}. Skipping evaluation.")
        return

    # Data processing logic copied from the notebook
    node_df_attacked_easy['time'] = pd.to_datetime(node_df_attacked_easy['time'])
    node_df_attacked_easy = node_df_attacked_easy.set_index(['node_id', 'time']).sort_index()
    idx = pd.IndexSlice

    end_time = '2018-11-03 00:00:00'
    new_end_time = '2018-11-04 00:00:00'
    times = node_df_attacked_easy.index.get_level_values("time").unique()
    try:
        start_time_ind = times.get_loc(end_time)
        start_time = times[start_time_ind - 30 + 2].strftime('%Y-%m-%d %H:%M:%S')
    except KeyError:
        start_time = '2018-11-01 00:00:00'

    n_nodes = 5
    node_df_attacked_easy_test = node_df_attacked_easy.loc[idx[0:n_nodes, start_time:new_end_time], :]

    ground_truth_df_easy['time'] = pd.to_datetime(ground_truth_df_easy['time'], yearfirst=True)
    ground_truth_df_easy = ground_truth_df_easy.set_index('time').sort_index()
    ground_truth_df_easy_test = ground_truth_df_easy[start_time:new_end_time]

    node_df_normal = node_df_normal.rename(columns={'node': 'node_id'})
    node_df_normal['time'] = pd.to_datetime(node_df_normal['time'])
    node_df_normal = node_df_normal.set_index(['node_id', 'time']).sort_index()
    node_df_normal_test = node_df_normal.loc[idx[0:n_nodes, start_time:new_end_time], :]

    market_df_easy["time"] = pd.to_datetime(market_df_easy["time"], yearfirst=True)
    market_df_easy = market_df_easy.set_index("time").sort_index()
    market_df_easy_test = market_df_easy[start_time:new_end_time]

    if 'Unnamed: 0' in node_df_attacked_easy_test.columns:
        node_df_attacked_easy_test = node_df_attacked_easy_test.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in ground_truth_df_easy_test.columns:
        ground_truth_df_easy_test = ground_truth_df_easy_test.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in market_df_easy_test.columns:
        market_df_easy_test = market_df_easy_test.drop(columns=['Unnamed: 0'])

    # Reset indices before passing to evaluation
    node_df_attacked_easy_test = node_df_attacked_easy_test.reset_index()
    ground_truth_df_easy_test = ground_truth_df_easy_test.reset_index()
    node_df_normal_test = node_df_normal_test.reset_index()
    
    run_evaluation(teacher_agent, node_df_attacked_easy_test, node_df_normal_test, ground_truth_df_easy_test, market_df_easy_test, step_size=1) 
    metrics_df = calculate_metrics(log_path=args.log_path)

if __name__ == "__main__":
    main()

