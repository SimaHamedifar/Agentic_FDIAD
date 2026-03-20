import re
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime

def parse_llm_json(text: str):
    # Remove markdown fencing
    cleaned = re.sub(r"```json|```", "", text).strip() # json.load() failed because of backticks
    
    # Try direct JSON loading
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def to_json_safe(x):
    """Convert any object (timestamp, tensor, ndarray, tuple, dict) to JSON-serializable form."""

    if x is None:
        return None

    # 🔥 FIX: Pandas or NumPy timestamps
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x).isoformat()

    # python datetime
    if isinstance(x, datetime):
        return x.isoformat()

    # torch tensor -> list
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()

    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # tuple -> list
    if isinstance(x, tuple):
        return [to_json_safe(i) for i in x]

    # list -> sanitize
    if isinstance(x, list):
        return [to_json_safe(i) for i in x]

    # dict -> ensure keys = strings
    if isinstance(x, dict):
        return {str(k): to_json_safe(v) for k, v in x.items()}

    # DataFrame -> dict of lists
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        # Ensure index names don't conflict with columns
        new_names = []
        for i, name in enumerate(df.index.names):
            default_name = name if name is not None else (f"level_{i}" if df.index.nlevels > 1 else "index")
            while default_name in df.columns:
                default_name = f"_{default_name}"
            new_names.append(default_name)
        df.index.names = new_names
        
        df = df.reset_index()
        # convert index and values
        return {col: [to_json_safe(v) for v in df[col].tolist()] for col in df.columns}

    # simple JSON-serializable values
    if isinstance(x, (int, float, str, bool)):
        return x

    # fallback: stringify everything else
    return str(x)
