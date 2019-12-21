import numpy as np
from numpy import ndarray
from scipy.stats import truncnorm

def sample_truncated_gaussian(mean: ndarray, y, size=None):
    """
    Sample a vector of truncated Gaussians z with means
    mean[i] and in the support z[i]y[i] > 0.
    """
    dim = len(mean)
    a = -mean
    b = -mean
    a[y < 0] = -np.inf
    b[y > 0] = np.inf
    if size is not None:
        size = (size, dim)
    z = truncnorm.rvs(a, b, loc=mean, size=size)
    return z
