from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.intake_agent import intake_node
from src.agents.fraud_agent import fraud_scoring_node
from src.agents.cost_policy_agent import cost_policy_node
from src.agents.validation_agent import validation_node

def create_fraud_screening_graph():
    """
    Builds the LangGraph workflow managing conditional routing,
    specialist agents, validation review, and at most one revision loop.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("intake", intake_node)
    workflow.add_node("fraud", fraud_scoring_node)
    workflow.add_node("policy", cost_policy_node)
    workflow.add_node("validation", validation_node)
    
    workflow.set_entry_point("intake")

    workflow.add_conditional_edges(
        "intake",
        lambda state: state.get("route"),
        {
            "fraud": "fraud",
            "human_review": END
        }
    )
    workflow.add_edge("fraud", "policy")
    
    workflow.add_edge("policy", "validation")

    workflow.add_conditional_edges(
        "validation",
        lambda state: state.get("route"),
        {
            "policy": "policy",       
            "human_review": END,      
            "end": END               
        }
    )
    
    return workflow.compile()