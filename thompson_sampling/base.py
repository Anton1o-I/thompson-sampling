from numpy.random import beta, gamma, binomial, exponential
from functools import partial
from pandas import Series, DataFrame
import operator
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


class BasePrior:
    def __init__(self):
        self.priors = {}

    def _param_calculator(self, *args):
        return self

    def add_one(
        self, mean: int, variance: int, effective_size: int, label: str
    ) -> dict:
        """
        Allows for individual priors to be specified and added to the priors list
        """

        new_prior = {label: self._param_calculator(mean, variance, effective_size)}
        self.priors.update(new_prior)
        return self

    def add_multiple(
        self, means: Series, variances: Series, sample_sizes: Series, labels: Series
    ) -> List[dict]:
        """
        Allows for a group of priors to be specified at once

        information: DataFrame
        """
        pass


class BaseThompsonSampling:
    def __init__(self, arms: int = None, priors: BasePrior = None, labels: list = None):
        self._avail_posteriors = {"beta": partial(beta), "gamma": partial(gamma)}
        if arms is None and priors is None:
            raise ValueError("Must have either arms or priors specified")
        if priors:
            self.posteriors = priors.priors
        elif arms:
            self.posteriors = {
                (f"{labels[i]}" if labels else f"option{i+1}"): self._default.copy()
                for i in range(arms)
            }

    def _sample_posterior(self, size: int = None, key: str = None):
        return self._avail_posteriors[self._posterior](
            size=size, **self.posteriors[key]
        )

    def pull_arm(self):
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

