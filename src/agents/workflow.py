from langgraph.graph import StateGraph, START, END

from src.agents.state import FDIState
from src.agents.nodes import (
    input_node,
    GATNode,
    gat_interpreter,
    observation_translator,
    merger,
    LLM_node,
    ParquetLoggerNode
)

def build_teacher_agent(gat_model_path="checkpoints/best_gatmarket_SixNode_model.pth", log_path="teacher_experience_new/data.parquet"):
    """
    Builds and compiles the LangGraph workflow for the teacher agent.
    """
    workflow = StateGraph(FDIState)
    
    # Initialize stateful nodes
    gat_node = GATNode(file_path=gat_model_path)
    gat_interp = gat_interpreter()
    logger = ParquetLoggerNode(path=log_path)
    
    # Add nodes
    workflow.add_node("input_node", input_node)
    workflow.add_node("gat", gat_node)
    workflow.add_node("gat_interpreter", gat_interp)
    workflow.add_node("observation_translator", observation_translator)
    workflow.add_node("merger", merger)
    workflow.add_node("LLM_node", LLM_node)
    workflow.add_node("logger", logger)
    
    # Add edges
    workflow.add_edge(START, "input_node")
    workflow.add_edge("input_node", "gat")
    workflow.add_edge("gat", "gat_interpreter")
    workflow.add_edge("gat_interpreter", "observation_translator")
    workflow.add_edge("observation_translator", "merger")
    workflow.add_edge("merger", "LLM_node")
    workflow.add_edge("LLM_node", "logger")
    workflow.add_edge("logger", END)
    
    # Compile the workflow
    teacher_agent = workflow.compile()
    
    return teacher_agent

