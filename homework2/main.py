import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

plt.style.use("seaborn")


iris_data = load_iris()
X = iris_data.data

print("Iris data shape:", X.shape)

from diag_em import gaussian_mixture_diagonal

num_classes = 3

sub_feats = [1, 3]

X_sub = X[:, sub_feats]
P, mus, sigmas = gaussian_mixture_diagonal(X_sub, num_classes, max_iters=10)

print("Class probabilities (diagonal mixture):")
print(P)

gm = GaussianMixture(n_components=num_classes)
gm.fit(X_sub)



def plot_ellipse(mean: ndarray, cov: ndarray, ax: plt.Axes, n_std=1., alpha=.2):
    from matplotlib.patches import Ellipse
    from scipy.linalg import eigh
    # Get eigendecomposition of the ellipse
    print("Cov matrix:\n", cov)
    lbda, v = eigh(cov)
    stdevs = n_std * (lbda ** .5)
    angle = np.degrees(np.arctan2(*v[:, 0]))
    # rmk: width, height are the diameters, not the radii
    ell = Ellipse(xy=mean, width=2*stdevs[0], height=2*stdevs[1],
                  angle=angle,
                  edgecolor='k',
                  linewidth=1.,
                  alpha=alpha,
                  zorder=1)
    ax.add_patch(ell)


# Number of stds for plotting
n_std = 1


fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(*X_sub.T, s=15, alpha=.6, edgecolors='k', lw=.8)

for index, cent in enumerate(mus):
    print("Class %d" % (index+1))
    ax.scatter(*cent, marker="o",
               zorder=2,
               label="Class %d centroid" % index,
               edgecolor='k')

    plot_ellipse(cent, sigmas[index], ax, n_std=n_std)

ax.legend()
ax.set_title("Diagonal mixture model (features %s)" % sub_feats)
fig.tight_layout()

plt.show()


## Full covariance model
print("Full covariance model:")

mus_full = gm.means_
sigmas_full = gm.covariances_

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(*X_sub.T, s=15, alpha=.6, edgecolors='k', lw=.8)

for index, cent in enumerate(mus_full):
    print("Class %d" % index)
    ax.scatter(*cent, marker="o",
               zorder=2,
               label="Class %d centroid" % (index+1),
               edgecolor='k')

    plot_ellipse(cent, sigmas_full[index], ax, n_std=n_std)

ax.legend()
ax.set_title("Full Gaussian mixture model (features %s)" % sub_feats)
fig.tight_layout()

plt.show()
