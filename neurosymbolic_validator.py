import networkx as nx
from typing import List, Tuple, Dict

class ClinicalKnowledgeGraph:
    def __init__(self):
        self.kg = nx.DiGraph()
        self._build_mock_umls_graph()

    def _build_mock_umls_graph(self):
        """
        Builds a mock Knowledge Graph representing established medical literature.
        In reality, this would query UMLS or SNOMED-CT APIs.
        """
        # Established directed relationships: Cause -> Effect
        edges = [
            ('age', 'sysBP'),
            ('male', 'TenYearCHD'),
            ('cigsPerDay', 'sysBP'),
            ('sysBP', 'TenYearCHD'),
            ('glucose', 'diabetes'),
            ('diabetes', 'TenYearCHD'),
            ('prevalentHyp', 'TenYearCHD'),
            ('BPMeds', 'sysBP')
        ]
        self.kg.add_edges_from(edges)

    def validate_causal_dag(self, data_driven_edges: List[Tuple[str, str]]) -> Dict:
        """
        Cross-validates edges found by the CausalLearn PC algorithm against the Knowledge Graph.
        """
        validated_edges = []
        hallucinated_edges = []
        novel_discoveries = []

        for source, target in data_driven_edges:
            if self.kg.has_edge(source, target):
                validated_edges.append((source, target))
            else:
                # Check if it contradicts established knowledge (e.g., reversed direction)
                if self.kg.has_edge(target, source):
                    hallucinated_edges.append((source, target))
                else:
                    # An edge not in the KG, potentially a novel insight from the data
                    novel_discoveries.append((source, target))

        return {
            "Validated (Match Literature)": validated_edges,
            "Hallucinations (Contradict Literature)": hallucinated_edges,
            "Novel Discoveries (Not in Literature)": novel_discoveries
        }
