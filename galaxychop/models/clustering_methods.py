import sklearn
import pandas as pd

import itertools
from typing import Optional, List
import numpy as np

class ClusteringMethod:
    pass
class HierarchicalClustering(ClusteringMethod):
    """
    Hierarchical clustering method.

    This class is used to cluster the start data of a galaxy.

    Parameters
    ----------
    - comp: Components
        The components of a galaxy.
    """
    def __init__(self, comp):
        self._comp = comp
        self._has_probs = comp.probabilities is not None
        
    def run(self, n_clusters: int = 4, linkage: str = 'ward', attributes: Optional[List[int]] = None):
        attribute_indeces = range(0, len(self._comp.attributes)) if attributes is None else attributes
        galaxy_data = self._prepare_data(attribute_indeces)
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clustering.fit(galaxy_data)
        
        return np.array(clustering.labels_)
    
    def run_all(self):
        """
        Run the clustering on all the possible combinations with the 'normalized_star_energy',
        'eps', and 'eps_r' attributes, and from 2 to 4 clusters.
        """
        all_attributes = self._comp.attributes
        results = []
        index = []
        x = 0
        for _, amount_attributes in enumerate(range(1, len(all_attributes)+1)):
            combined_attributes = list(itertools.combinations(enumerate(all_attributes), amount_attributes))
            for attributes in combined_attributes:
                index.append(attributes)
                results.append([])
                for _, n_clusters in enumerate([2,3,4]):
                    attribute_indices = [attribute[0] for attribute in attributes]
                    results[x].append(self.run(n_clusters = n_clusters, attributes = attribute_indices))
                x=x+1
        
        index = list(map(lambda a: str(tuple(map(lambda b: b[1], a))), index))

        return pd.DataFrame(results,
            index=pd.Index(index, name='Attributes'),
            columns=pd.Index([2,3,4], name='n_clusters')
        )
    
    def _prepare_data(self, attribute_indices: List[int]):
        data = [[t[i] for i in attribute_indices] for t in self._comp.x_clean]
        attributes = [self._comp.attributes[i] for i in attribute_indices]
        
        galaxy_data = pd.DataFrame(data, columns = attributes)
        return galaxy_data