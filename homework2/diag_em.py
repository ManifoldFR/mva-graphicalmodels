import numpy as np
from numpy import ndarray
from scipy.stats import multivariate_normal as MVN

from sklearn.cluster import k_means


def gaussian_mixture_diagonal(X: ndarray, num_class: int, max_iters: int = 100):
    """
    
    X: 2D array of shape (n_samples, n_features)
    num_classes: number of classes
    """
    n_samples = X.shape[0]
    dim = X.shape[1]
    ## First initialize the parameters using KMeans
    mus, labels, *_ = k_means(X, num_class)
    
    P = np.empty((num_class, 1))  # shape (num_class, 1)
    for k in range(num_class):
        P[k, 0] = np.sum(labels == k) / n_samples

    
    sigmas = np.stack([
        np.diagflat(np.ones(dim))
        for k in range(num_class)
    ])

    def conditional_probabilities(X, mus, sigmas):
        """Compute vector of p(X|k) for all k
        
        Returns: array of shape (num_class, n_samples)
        """
        return np.stack([
            MVN.pdf(X, mean=mus[k], cov=sigmas[k])
        for k in range(num_class)])

    diag_idx = np.diag_indices(dim)
    q = np.empty((num_class, n_samples))

    for m in range(max_iters):
        # EXPECTATION STEP
        ## compute the posterior probabilities
        cond_ = conditional_probabilities(X, mus, sigmas)  # p(X|k) (num_class, n_samples)
        q[:] = P * cond_  # shape (num_class, n_samples)
        q[:] /= np.sum(q, axis = 0, keepdims=True)
        w = np.sum(q, axis = 1, keepdims=True)  # weights, shape (num_class, 1)
        
        # MAXIMIZATION STEP
        ## update probabilities
        P[:, 0] = np.mean(q, axis = 1)
        
        for k in range(num_class):
            # X of shape (n_samples, dim)
            # q[k] of shape n_samples -> right operation is X.T @ q[k]
            mus[k][:] = X.T @ q[k] / w[k]
            diagos_ = (X - mus[k]) ** 2  # shape (n_samples, dim)
            qk = q[k, :, None]  # shape (n_samples, 1)
            new_sigma = np.sum(qk * diagos_, axis=0) / w[k]
            sigmas[k][diag_idx] = new_sigma

    return P, mus, sigmas
