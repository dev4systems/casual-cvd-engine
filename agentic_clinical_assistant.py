# agentic_clinical_assistant.py
import pandas as pd
import numpy as np
from typing import Dict, List, TypedDict, Annotated, Union
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Define the Clinical State ---
class ClinicalState(TypedDict):
    """
    State representing the clinical context of a patient.
    Note: patient_data is stored as a list of dicts for serializability.
    """
    patient_data: List[Dict]
    survival_probability: float
    # List of dictionaries describing proposed changes
    proposed_treatment: List[Dict]
    # Physician feedback
    physician_approval: bool
    physician_notes: str
    # Internal flag to handle node transitions
    risk_level: str  # 'Optimal' or 'Intervention_Required'

# --- 2. Mock Digital Twin Model (For Standalone Functionality) ---
class MockDigitalTwin:
    """
    A mock class simulating a Cox Proportional Hazards or DeepSurv model.
    """
    def predict_survival_function(self, data: pd.DataFrame):
        # Returns a mock survival curve
        # Index represents years, columns are patient IDs
        years = list(range(1, 11))
        # Logic: higher sysBP or smoking lowers survival
        base = 0.95
        penalty = 0.0
        if data['sysBP'].iloc[0] > 140: penalty += 0.15
        if data['cigsPerDay'].iloc[0] > 10: penalty += 0.10
        
        curve = [max(0.1, base - (penalty * (y/10))) for y in years]
        return pd.DataFrame(curve, index=years, columns=[0])

# --- 3. Define the Nodes ---

def monitoring_node(state: ClinicalState):
    """
    Evaluates the patient's baseline 10-year survival probability.
    """
    print("\n[Node: Monitoring] Evaluating baseline risk...")
    model = MockDigitalTwin()
    # Convert list of dict back to DataFrame for processing
    df = pd.DataFrame(state['patient_data'])
    survival_func = model.predict_survival_function(df)
    
    # Get survival probability at year 10
    prob_10_year = float(survival_func.loc[10].values[0])
    
    risk_level = "Optimal" if prob_10_year >= 0.80 else "Intervention_Required"
    
    return {
        "survival_probability": prob_10_year,
        "risk_level": risk_level
    }

def reasoning_node(state: ClinicalState):
    """
    Autonomously simulates counterfactual interventions and formulates a plan.
    """
    print("[Node: Reasoning] Risk below threshold. Simulating counterfactuals...")
    model = MockDigitalTwin()
    # Convert list of dict back to DataFrame
    patient = pd.DataFrame(state['patient_data'])
    
    # Intervention Scenarios
    scenarios = [
        {"desc": "Target Systolic BP to 120", "changes": {"sysBP": 120.0}},
        {"desc": "Complete Smoking Cessation", "changes": {"cigsPerDay": 0.0}},
        {"desc": "Aggressive Combined Therapy", "changes": {"sysBP": 120.0, "cigsPerDay": 0.0}}
    ]
    
    best_outcome = 0.0
    best_plan = []
    
    for scenario in scenarios:
        sim_data = patient.copy()
        for col, val in scenario['changes'].items():
            sim_data[col] = val
        
        sim_prob = float(model.predict_survival_function(sim_data).loc[10].values[0])
        
        if sim_prob > best_outcome:
            best_outcome = sim_prob
            best_plan = [scenario]

    print(f"[Node: Reasoning] Best simulated outcome: {best_outcome:.2f}")
    
    return {
        "proposed_treatment": best_plan
    }

def hitl_node(state: ClinicalState):
    """
    A placeholder node for Physician Review.
    The graph will interrupt BEFORE this node to allow for Human-In-The-Loop.
    """
    print("[Node: HITL] Physician review complete.")
    return state

# --- 4. Define the Routing Logic ---

def should_intervene(state: ClinicalState):
    """
    Conditional edge logic based on risk level.
    """
    if state["risk_level"] == "Optimal":
        print("[Edge] Risk is acceptable. Ending workflow.")
        return "end"
    else:
        print("[Edge] Risk threshold breached. Moving to reasoning.")
        return "continue"

# --- 5. Build the LangGraph State Machine ---

workflow = StateGraph(ClinicalState)

# Add Nodes
workflow.add_node("monitor", monitoring_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("hitl_review", hitl_node)

# Define Edges
workflow.set_entry_point("monitor")

workflow.add_conditional_edges(
    "monitor",
    should_intervene,
    {
        "continue": "reasoning",
        "end": END
    }
)

workflow.add_edge("reasoning", "hitl_review")
workflow.add_edge("hitl_review", END)

# Configure Checkpointer for context retention
memory = MemorySaver()

# Compile the Graph with HITL interrupt before the physician review
clinical_agent = workflow.compile(
    checkpointer=memory,
    interrupt_before=["hitl_review"]
)

# --- 6. Execution Example ---
if __name__ == "__main__":
    # Mock high-risk patient data as a list of dicts for serialization
    patient_data = [{
        'sysBP': 165.0,
        'cigsPerDay': 25.0,
        'age': 55,
        'male': 1,
        'glucose': 110.0
    }]
    
    initial_state = {
        "patient_data": patient_data,
        "survival_probability": 0.0,
        "proposed_treatment": [],
        "physician_approval": False,
        "physician_notes": "",
        "risk_level": "Optimal"
    }

    thread_config = {"configurable": {"thread_id": "patient_001"}}

    # Start the graph
    print("--- Starting Clinical Assistant Workflow ---")
    for event in clinical_agent.stream(initial_state, thread_config):
        for node, values in event.items():
            print(f"Update from {node}")

    # Check if we are interrupted
    state = clinical_agent.get_state(thread_config)
    if state.next:
        print("\n!!! ACTION REQUIRED: Graph paused for Physician Review !!!")
        print(f"Current survival probability: {state.values['survival_probability']:.2%}")
        print(f"Proposed Treatment: {state.values['proposed_treatment']}")
        
        # Simulating User (Physician) Input
        notes = "Approved combined therapy. Patient also advised on exercise."
        
        # Resume graph with physician input
        print("\n--- Resuming Workflow with Physician Input ---")
        clinical_agent.update_state(
            thread_config,
            {"physician_approval": True, "physician_notes": notes},
            as_node="hitl_review" 
        )
        
        # Resuming the stream
        for event in clinical_agent.stream(None, thread_config):
             for node, values in event.items():
                print(f"Final update from {node}")

    print("\n--- Workflow Complete ---")
