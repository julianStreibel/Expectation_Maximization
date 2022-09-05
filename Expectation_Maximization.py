import numpy as np
from scipy.stats import multivariate_normal


class Expectation_Maximization:
    def __init__(self, X: np.array, number_of_classes: int):
        self.X = X
        self.n, self.k = X.shape
        self.means = np.random.rand(
            number_of_classes, self.k)
        self.means -= 0.5
        self.means = np.multiply(self.means, X.max(axis=0))
        self.sigmas = np.zeros((number_of_classes, self.k, self.k))
        idx = np.arange(self.k)
        self.sigmas[:, idx, idx] = 1
        self.prior = np.ones((number_of_classes)) / number_of_classes
        self.likelihood = np.zeros((number_of_classes, self.n))
        self.posterior = np.zeros((number_of_classes, self.n))

    def expectation_step(self):
        for i, (m, s) in enumerate(zip(self.means, self.sigmas)):
            # TODO: save multi_normals instead of mean and sigma
            self.likelihood[i] = multivariate_normal(m, s).pdf(self.X)
        joined = np.multiply(self.likelihood.T, self.prior).T
        evidence = joined.sum(axis=0)
        self.posterior = joined / evidence

    def maximization_step(self):
        posterior_sum = self.posterior.sum(axis=1)
        self.means = self.posterior @ self.X / posterior_sum.reshape(-1, 1)
        for i, _ in enumerate(self.sigmas):  # TODO: vectorize
            distance_to_mean = self.X - self.means[i]
            self.sigmas[i] = distance_to_mean.T @ np.diag(
                self.posterior[i]) @ distance_to_mean / posterior_sum[i]

    def step(self):
        self.expectation_step()
        self.maximization_step()
