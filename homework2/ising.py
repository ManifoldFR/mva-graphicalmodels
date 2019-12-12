"""
Ising model.
"""
import numpy as np
from numpy import ndarray

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
        print("edge psis:", base_psi_edge_log)
        
        width = self.width
        # array of all possible combinations
        psi_tree_log = np.empty((2,) * width)
        psi_edge_tree_log = np.empty((2,) * width)
        
        it = np.nditer(
            psi_tree_log,
            flags=['multi_index'], op_flags=['writeonly'])
        while not(it.finished):
            idx = it.multi_index
            it[0] = np.sum([base_psi_log[j] for j in idx])
            # len(idx) = width
            it[0] += np.sum([
                base_psi_edge_log[idx[k], idx[k+1]]
                for k in range(width-1)
            ])
            
            psi_edge_tree_log[idx] = np.sum([
                base_psi_edge_log
            ])
            it.iternext()

        self.psi_tree_log = psi_tree_log.ravel()  # flatten to 1D array

        # make list of pointers to the same tree psi
        psis_chain = [np.exp(self.psi_tree_log) for _ in range(self.height)]
        
        
        chain = UndirectedChain(psis_chain, )


if __name__ == "__main__":
    width = 10
    height = 100
    alpha = 0.5
    beta = 1.
    
    grid = Ising(width, height, alpha, beta)
    
    grid.make_junction_tree()

    print(grid.psi_tree_log.shape)

