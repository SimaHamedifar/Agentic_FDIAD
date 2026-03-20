# FDIA Detector Agent

This repository contains a reinforcement learning-based False Data Injection Attack (FDIA) detection agent for Home Energy Management System (HEMS) data, utilizing GAT-LSTM and RL algorithms. The agent integrates graph neural network methodologies within a LangGraph-orchestrated workflow.

## Project Structure

The codebase is organized into several modules for readability and separation of concerns:

- `src/agents/state.py`: Defines the `FDIState` object that flows through the LangGraph and model `Config`.
- `src/agents/nodes.py`: Contains the individual LangGraph nodes (`input_node`, `GATNode`, `gat_interpreter`, `observation_translator`, `merger`, `LLM_node`, `ParquetLoggerNode`).
- `src/agents/workflow.py`: Constructs, connects, and compiles the AI agent LangGraph.
- `src/utils/helpers.py`: Contains general helper functions like JSON-safe converters.
- `src/utils/evaluation.py`: Houses the evaluation logic, retry mechanisms, and metrics calculation.
- `run.py`: The main CLI entry point.

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the detector agent evaluation, execute:

```bash
python run.py --model_path checkpoints/best_gatmarket_SixNode_model.pth --log_path detector_experience_new/data.parquet --data_dir Utils/Data
```

You can pass `-h` or `--help` to see the available command-line arguments.
