# %%
from plotting import plot_multiple_bi_gaussians
from sklearn.datasets import make_blobs
import numpy as np

from Expectation_Maximization import Expectation_Maximization

# %%
n_classes = 6

data, y = make_blobs(
    n_features=2,
    centers=n_classes,
    cluster_std=1,
    n_samples=100,
)
# %%
plot_multiple_bi_gaussians(
    [data[y == i].mean(axis=0) for i in range(n_classes)],
    [np.cov(data[y == i].T) for i in range(n_classes)],
    data,
    y
)

# %%
em = Expectation_Maximization(data, n_classes)

# %%
for i in range(56):
    if i % 5 == 0:
        plot_multiple_bi_gaussians(
            em.means,
            em.sigmas,
            data,
            y
        )
    em.step()
# %%
