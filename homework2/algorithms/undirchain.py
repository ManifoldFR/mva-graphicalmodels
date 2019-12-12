"""
Algorithm for computing the marginals in an undirected chain graph.
"""
import numpy as np
from numpy import ndarray
from typing import List
from math import log
from scipy.special import logsumexp


def logdotexp(log_A: ndarray, log_x: ndarray):
    """Perform the matrix-vector product A.x in log-scale"""
    return logsumexp(log_A + log_x, axis=1)
    


class UndirectedChain:
    """Undirected chain graphical model."""
    
    def __init__(self, log_psis: List[ndarray], log_psi_edges: List[ndarray]):
        """
        Args
            psis: List[array]. psis[i] is the log i-th factor
            psis: List[array]. psi_edges[i] is the log i-th factor
        """
        super().__init__()
        
        self.length: int = len(log_psis)
        print("Chain length:", self.length)
        self.psis: List[ndarray] = log_psis
        self.psi_edges: List[ndarray] = log_psi_edges

    def log_proba(self, multi_index: List[int], normalized: bool = False):
        """Evaluate the log-probability distribution of :math:`(X_1,\ldots,X_n)` at the multiindex."""
        res = 0.
        for i in range(self.length):
            x = multi_index[i]  # index of x_i
            res += self.psis[x]
            if i < self.length-1:
                y = multi_index[i+1]
                res += self.psi_edges[y]
        return res
    
    def compute_marginal(self, index: int):
        """Propagate.
        
        Args
            index: the idx of the variable we want to get the marginal of
        """
        ## Forward propagation
        mu_fwd = 0.
        for k in range(index):
            psi_i = self.psis[k]
            psi_ip1 = self.psi_edges[k]
            mu_fwd = logdotexp(psi_i + mu_fwd, psi_ip1.T)
        
        ## Backward propagation
        mu_bwd = 0.
        for k in range(self.length-1, index, -1):
            psi_i = self.psis[k]
            psi_im1i = self.psi_edges[k-1]
            mu_bwd = logdotexp(psi_i + mu_bwd, psi_im1i)
        
        psi_idx = self.psis[index]
        
        return np.exp(mu_fwd + psi_idx + mu_bwd)

