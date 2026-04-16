import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

def run_causal_discovery():
    # 1. Load the cleaned training data
    df = pd.read_csv("cleaned_training_data.csv")
    
    # We'll use a subset of variables for the DAG to keep it readable and focused
    # PC Algorithm works best with continuous/ordinal data
    data = df.to_numpy()
    labels = df.columns.tolist()
    
    print("Running PC Algorithm for Causal Discovery...")
    # 2. Run PC Algorithm
    # alpha is the significance level for independence tests
    cg = pc(data, alpha=0.05, labels=labels)
    
    # 3. Identify Direct Parents of TenYearCHD
    target_idx = labels.index('TenYearCHD')
    adj_matrix = cg.G.graph
    
    parents = []
    for i in range(len(labels)):
        # In causal-learn adjacency: 
        # -1: no edge, 1: directed edge i->j, -1: j->i, 2: i--j (undirected)
        # However, the exact mapping depends on the graph type.
        # We look for nodes that have an edge pointing to TenYearCHD.
        if adj_matrix[i, target_idx] == 1 or adj_matrix[i, target_idx] == 2:
            parents.append(labels[i])
            
    print("\n--- Causal Discovery Results ---")
    print(f"Variables identified as potential causal drivers (parents/neighbors) of TenYearCHD:")
    for p in parents:
        print(f"- {p}")
        
    # 4. Export the DAG visualization to a dot file
    pyd = GraphUtils.to_pydot(cg.G)
    pyd.write_png('causal_dag.png')
    print("\nDAG visualization saved as 'causal_dag.png'")

if __name__ == "__main__":
    run_causal_discovery()
