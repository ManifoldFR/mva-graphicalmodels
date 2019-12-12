"""
Ising model.
"""
import numpy as np
from numpy import ndarray
import numba

from algorithms.undirchain import UndirectedChain


class Ising:
    """Ising model with a junction tree.
    
    Attributes:
        width: width of the grid
        height: height of the grid
    """
    
    def __init__(self, width: int, height: int, alpha, beta):
        """
        Define the grid.
        
        Args:
            width (int): 
            height (int): 
            alpha (float)
            beta (float)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.psi_tree_log: ndarray = None


    def make_junction_tree(self):
        
        base_states = np.array([0., 1.])
        base_psi_log: ndarray = self.alpha * base_states  # shape (2,)
        
        base_psi_edge_log: ndarray = self.beta * (
            (base_states[None] == base_states[:, None]))
        # print("edge psis:", base_psi_edge_log)
        
        width = self.width
        # array of all possible combinations -- psi of the clusters
        shape = (2,) * width
        psi_tree_log = np.empty(shape)
        ## Make the array of factors between the clusters
        psi_edge_tree_log = np.empty(2 * shape)
        
        ndindex = np.ndindex(*shape)  # will be reused so collect
        
        for idx in ndindex:
            # indices j in idx run over the width of the grid
            # Invariant: len(idx) = width
            psi_tree_log[idx] = np.sum([base_psi_log[j] for j in idx])
            psi_tree_log[idx] += np.sum([
                base_psi_edge_log[idx[j], idx[j+1]] for j in range(width-1)
            ])
        
        _junction_edge_fill(psi_edge_tree_log, base_psi_edge_log)
        
        self.psi_tree_log = psi_tree_log.ravel()  # flatten to 1D array
        self.psi_edge_tree_log = psi_edge_tree_log.ravel()

        # make list of pointers to the same tree psi
        psis_chain = [self.psi_tree_log for _ in range(self.height)]
        psi_edges_chain = [self.psi_edge_tree_log for _ in range(self.height)]
        
        chain = UndirectedChain(psis_chain, psi_edges_chain)
        return chain

## Do the iteration waaay faster
@numba.njit
def _junction_edge_fill(psi_edge_tree_log, base_psi_edge_log):
    ndindex_edge = np.ndindex(*psi_edge_tree_log.shape)
    for idx in ndindex_edge:
        idx1 = idx[width:]
        idx2 = idx[:width]
        psi_edge_tree_log[idx] = 0.
        for j in range(width):
            psi_edge_tree_log[idx] += base_psi_edge_log[idx1[j], idx2[j]]
        

if __name__ == "__main__":
    width = 10
    height = 100
    alpha = 0.5
    beta = 1.4
    
    grid = Ising(width, height, alpha, beta)
    
    chain = grid.make_junction_tree()
    

