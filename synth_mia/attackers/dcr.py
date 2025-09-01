import numpy as np
from ..base import BaseAttacker
from scipy.spatial import cKDTree
from typing import Dict, Any, Tuple, List, Optional

class DCR(BaseAttacker):
    """
    Black Box Distance to Closest Record (DCR) attack. 
    Chen, D., Yu, N., Zhang, Y., and Fritz, M. Gan-leaks: A taxonomy of membership inference attacks againstgenerative models. 
    In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, CCS '20. ACM, October 2020. doi: 10.1145/3372297.3417238. 
    URL http://dx.doi.org/10.1145/3372297.3417238.
    """
    def __init__(self, distance_type=2):
        """Initialize DCR attacker.
        
        Args:
            distance_type: Type of distance metric (default: 2 for L2 norm)
        """
        super().__init__(distance_type=distance_type)
        self.name = "DCR"

    def find_nearest_neighbor_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Find the distance to the nearest neighbor in Y for each point in X.
        
        Args:
            X (np.ndarray): Query points
            Y (np.ndarray): Reference points
        
        Returns:
            np.ndarray: Array of distances to the nearest neighbor
        """
        tree = cKDTree(Y)
        distances, _ = tree.query(X, k=1, p=self.distance_type)
        return distances


    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the attack scores using efficient nearest neighbor search.
        
        Args:
            X_test (np.ndarray): Test data (member and non-member)
            synth (np.ndarray): Synthetic data
            ref (np.ndarray): Reference data (not used in this implementation)
        
        Returns:
            np.ndarray: Predicted scores
        """
        nearest_distances = self.find_nearest_neighbor_distances(X_test, synth)
        scores =  -(nearest_distances)
        return scores
