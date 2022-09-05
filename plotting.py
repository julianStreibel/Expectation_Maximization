import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

colors = np.random.rand(100, 3) * 0.5 + 0.25


def plot_bi_gauss(mean: np.array, sigma: np.array, color: np.array, ax, n_sigma=3):
    pearson = sigma[0, 1]/np.sqrt(sigma[0, 0] * sigma[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.

    for i in range(1, n_sigma + 1):
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=[*color, 1/(i + 5)])

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(sigma[0, 0]) * i

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(sigma[1, 1]) * i

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(*mean)

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)


def plot_multiple_bi_gaussians(means, sigmas, data, y):
    _, ax = plt.subplots(figsize=(6, 6))
    for i, (m, s) in enumerate(zip(means, sigmas)):
        plot_bi_gauss(m, s, colors[i], ax)
    ax.scatter(data[:, 0], data[:, 1], c=y)
    plt.show()
