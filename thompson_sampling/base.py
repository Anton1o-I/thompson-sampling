from numpy.random import beta, gamma, binomial, exponential
from functools import partial
import operator
import seaborn as sns
import matplotlib.pyplot as plt


class BaseThompsonSampling:
    def __init__(self):
        self._avail_posteriors = {"beta": partial(beta), "gamma": partial(gamma)}
        self._posterior = ""

    def _sample_posterior(self, size: int = None, key: str = None):
        return self._avail_posteriors[self._posterior](
            size=size, **self.posteriors[key]
        )

    def get_action(self):
        """
        Pull the slot machine arm

        Given the current posterior distributions this function will sample from
        the posterior and find the max theta of all the available options
        """

        theta_est = {}
        for key, _ in self.posteriors.items():
            theta_est[key] = self._sample_posterior(1, key)
        return max(theta_est.items(), key=operator.itemgetter(1))[0]

    def plot_posterior(self):
        plot_values = {
            key: self._sample_posterior(10000, key)
            for key, _ in self.posteriors.items()
        }

        for key, sim_array in plot_values.items():
            sns.distplot(sim_array, hist=False, label=key)
        plt.title("Posterior Distributions")
        plt.legend()
        plt.xlabel("Parameter Value")

