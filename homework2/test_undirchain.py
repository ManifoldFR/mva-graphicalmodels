"""
Tests for undirected chains.
"""
import numpy as np
from algorithms.undirchain import UndirectedChain


def test_independent_edges():
    dim1 = 3
    dim2 = 25
    dim3 = 2

    mid_arr = np.linspace(-4, 4, dim2)
    
    beta = 4.
    psis_= [
        np.ones((dim1)),
        np.exp(-mid_arr**2 / beta),  # Laplace distribution
        np.ones((dim3)),
    ]
    
    psis_[0][1] = 2.
    
    ## Independent edges
    psi_edges_ = [
        np.ones((dim1, dim2)),
        np.ones((dim2, dim3)),
    ]
    
    log_psis = [np.log(p) for p in psis_]
    log_psi_edges = [np.log(pe) for pe in psi_edges_]

    chain = UndirectedChain(log_psis, log_psi_edges)

    marginals = []
    for idx in range(chain.length):
        msg_test_ = chain.compute_marginal(idx)
        print("Marginal #%d:" % (idx+1), msg_test_)
        marginals.append(msg_test_)

    Z = msg_test_.sum()
    print("Partition function:", Z)

    real_Z = 84.81013344662627

    assert np.allclose(Z, real_Z)
    return marginals, Z


if __name__ == "__main__":
    from algorithms.undirchain import logdotexp
    
    lA = np.zeros((2,2))
    lA[0, 0] = np.log(2.)
    lA[0, 1] = -np.log(2.)
    lx = np.zeros(2)
    print("Data:\n", np.exp(lA), " and ", np.exp(lx))
    print("logsumexp product:", np.exp(logdotexp(lA, lx)))
    
    print("normal product:", np.exp(lA) @ np.exp(lx))

    print("Independent nodes:")
        
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn')
    marginals, Z = test_independent_edges()
    marginals = [m_/Z for m_ in marginals]
    n_marg = len(marginals)
    
    fig: plt.Figure = plt.figure(figsize=(8,3))
    for i in range(n_marg):
        fig.add_subplot(1, n_marg, i+1)
        p_ = marginals[i]
        if i == 1:
            x_ = np.linspace(-4, 4, len(p_))
        else:
            x_ = np.array(range(len(p_)))
        dx = x_[1]-x_[0]
        plt.bar(x_, p_, width=.85 * dx)
        plt.title("$p(x_%d)$" % (i+1))
    
    fig.tight_layout()
    plt.show()
