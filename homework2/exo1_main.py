import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

plt.style.use("seaborn")


# iris_data = load_iris()
# X = iris_data.data
# labels_real = iris_data.target
# BASE_FOLDER = "images"

data_arr = np.loadtxt("data.dat")
X = data_arr[:, :2]
labels_real = data_arr[:, 2]
BASE_FOLDER = "custom_data"

os.makedirs(BASE_FOLDER, exist_ok=True)


print("Data shape:", X.shape)

from diag_em import gaussian_mixture_diagonal

num_classes = 2

P, mus, sigmas, labels = gaussian_mixture_diagonal(X, num_classes, max_iters=10)
labels_diag = labels


print("Class probabilities (diagonal mixture):")
print(P)
print()

gm = GaussianMixture(n_components=num_classes)
gm.fit(X)
labels_full = gm.predict(X)

print("Class probabilities (full mixture)")
print(gm.weights_)
print()

km = KMeans(n_clusters=num_classes)
km.fit(X)
labels_kmean = km.predict(X)



def plot_ellipse(mean: ndarray, cov: ndarray, ax: plt.Axes, n_std=1., alpha=.2, facecolor=None):
    from matplotlib.patches import Ellipse
    from scipy.linalg import eigh
    # Get eigendecomposition of the ellipse
    lbda, v = eigh(cov)
    stdevs = n_std * (lbda ** .5)
    angle = np.degrees(np.arctan2(*v[:, 0][::-1]))
    # rmk: width, height are the diameters, not the radii
    ell = Ellipse(xy=mean, width=2*stdevs[0], height=2*stdevs[1],
                  angle=angle,
                  edgecolor='k',
                  facecolor=facecolor,
                  linewidth=1.,
                  alpha=alpha,
                  zorder=1)
    ax.add_patch(ell)


def plot_scatter(X, labels_real, ax: plt.Axes):
    n_real_class = len(np.unique(labels_real))
    colors = []
    for idx_class, lbl in enumerate(np.unique(labels_real)):
        sub_X = X[labels_real == lbl]
        sc_ = ax.scatter(*sub_X.T, s=15, zorder=2,
                         label="Real label %s" % lbl,
                         edgecolors='k', lw=.8)



if __name__ == '__main__':
    # Number of stds for plotting
    n_std = 2


    from itertools import combinations
    dim = X.shape[1]
    feature_pairs = list(combinations(range(dim), 2))
    
    print("No. of feature pairs:", len(feature_pairs))
    
    print("Diagonal covariance model:")
    
    # nrows = 2
    # ncols = 3
    # figsize = (12, 5)
    
    nrows = 1
    ncols = 1
    figsize = (5, 4)
    
    
    marker = "v"
    
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]
    
    for j, sub_feats in enumerate(feature_pairs):
        X_sub = X[:, sub_feats]
        ax = axes[j]

        plot_scatter(X_sub, labels_real, ax)
        
        for index, cent in enumerate(mus):
            cent = cent[list(sub_feats)]
            sc_ = ax.scatter(*cent, marker=marker,
                    zorder=3,
                    label="Class %d" % (index+1),
                    edgecolor='w', s=100)

            cov = sigmas[index][np.ix_(sub_feats, sub_feats)]
            col = sc_.get_facecolor()[0]
            plot_ellipse(cent, cov, ax, n_std=n_std, facecolor=col)

        ax.legend(facecolor='white')
        ax.set_title("Diagonal mixture model (features %s)" % list(sub_feats))
    fig.tight_layout()
    
    fig.savefig(BASE_FOLDER+'/diag_em_K%d.pdf' % num_classes)

    plt.show()


    ## Full covariance model
    print("Full covariance model:")

    mus_full = gm.means_
    sigmas_full = gm.covariances_

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]
    
    for j, sub_feats in enumerate(feature_pairs):
        X_sub = X[:, sub_feats]
        ax = axes[j]

        colors = plot_scatter(X_sub, labels_real, ax)

        for index, cent in enumerate(mus_full):
            cent = cent[list(sub_feats)]
            sc_ = ax.scatter(*cent, marker=marker,
                    zorder=3,
                    label="Class %d" % (index+1),
                    edgecolor='w', s=100)
            cov = sigmas_full[index][np.ix_(sub_feats, sub_feats)]
            col = sc_.get_facecolor()[0]
            plot_ellipse(cent, cov, ax, n_std=n_std,
                         facecolor=col)

        ax.legend(facecolor='white')
        ax.set_title("Full Gaussian mixture model (features %s)" % list(sub_feats))
    fig.tight_layout()
    
    fig.savefig(BASE_FOLDER+'/full_em_K%d.pdf' % num_classes)

    plt.show()
    
    
    ## K-Means
    print("K-means:")
    
    centroids = km.cluster_centers_
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]
    
    for j, sub_feats in enumerate(feature_pairs):
        X_sub = X[:, sub_feats]
        ax = axes[j]

        colors = plot_scatter(X_sub, labels_real, ax)

        for index, cent in enumerate(centroids):
            cent = cent[list(sub_feats)]
            ax.scatter(*cent, marker=marker,
                    zorder=3,
                    label="Class %d" % (index+1),
                    edgecolor='w', s=100)

        ax.legend(facecolor='white')
        ax.set_title("K-means (features %s)" % list(sub_feats))
    fig.tight_layout()

    fig.savefig(BASE_FOLDER+'/kmeans_K%d.pdf' % num_classes)

    plt.show()
    
    




