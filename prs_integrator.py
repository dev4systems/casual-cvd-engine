import pandas as pd
import numpy as np

class PRSDataPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_integrate_prs(self) -> pd.DataFrame:
        """
        Loads the clinical dataset and merges/simulates a continuous Polygenic Risk Score (PRS).
        PRS is normally derived from GWAS (Genome-Wide Association Studies).
        """
        df = pd.read_csv(self.data_path)
        
        # SIMULATION: Generate a continuous PRS feature. 
        # In reality, this is ingested from genetic sequencing pipelines (e.g., PLINK).
        # We simulate it as a normal distribution where higher scores correlate slightly with CHD.
        np.random.seed(42)
        base_prs = np.random.normal(loc=0.0, scale=1.0, size=len(df))
        
        # Introduce a mild correlation with the target to simulate genetic predisposition
        chd_modifier = df['TenYearCHD'].values * 0.5 
        df['Polygenic_Risk_Score'] = base_prs + chd_modifier
        
        # Standardize the new feature
        df['Polygenic_Risk_Score'] = (df['Polygenic_Risk_Score'] - df['Polygenic_Risk_Score'].mean()) / df['Polygenic_Risk_Score'].std()
        
        print("Successfully integrated Polygenic Risk Score (PRS) into the feature space.")
        return df
