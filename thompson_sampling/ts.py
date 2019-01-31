from numpy import random, mean, percentile
import operator
import copy
import seaborn as sns
import matplotlib.pyplot as plt

# TODO Add exponential distribution
"""
Template for creating a Thompson Sampling Experiment

Currently supports the following distributions:
    - Binomial - For experiments with multiple groups and a binary outcome.
    - Poission - For experiments with multiple groups and a count based outcome.
    - Exponential - For experiments with multiple groups and a time based outcome.
    - Negative Binomial - For experiments with multiple groups and an outcome
        that measures number of successes before a failure.

"""


def negative_binomial_sim(posterior, label):
    p = random.beta(a=posterior[label]["alpha"], b=posterior[label]["beta"], size=1)
    n = 0
    success = 1
    while success == 1:
        success = random.binomial(n=1, p=p, size=1)
        n += 1
    return n


class ThompsonSampling:
    """
    Instantiates a new multi-armed bandit using the Thompson Sampling method.

    Attributes
    ----------
    likelihood
        Distribution name of the likelihood function to be used - currently
        available for "Poisson" and "Binomial" experiments
    priors
        Dictionary of dictionary for priors of each arm in the experiment.
        Example - {"OptionA":{"alpha":2, "beta":2 }, "OptionB":{"alpha":3, "beta":3}}

    """

    def __init__(self, likelihood: str, priors: dict):
        """
        Initializes the experiment by creating the posterior distribution dicts
        that will be utilized and updated throughout the course of the experiment.

        Currently supports experiments with Poisson, Exponential, Binomial and Negative Binomial likelihood functions,
        but more will be added. Requires dictionary with labels that signify the different
        experimental treatments being tested.

        """
        default_priors = {
            "Poisson": {"shape": 0.001, "rate": 0.001, "n": 0},
            "Binomial": {"alpha": 1, "beta": 1, "n": 0},
            "NegativeBinomial": {"alpha": 1, "beta": 1, "r": 1, "n": 0},
            "Exponential": {"shape": 0.001, "rate": 0.001, "n": 0},
        }
        if likelihood in default_priors.keys():
            self.likelihood = likelihood
        else:
            raise ValueError("Not a currently supported likelihood function")

        self.prior = {}
        for key, _ in priors.items():
            if priors[key]:
                self.prior[key] = priors[key]
            else:
                self.prior[key] = copy.deepcopy(default_priors[likelihood])

        self.posterior = self.prior

    def get_action(self):
        """
        Pull the Slot Machine Arm

        Using the most recent posterior distributions, it will randomly
        generate parameter value from posterior predictive and return label
        with maximum parameter value.

        Used to identify which option to select for next observation in the
        experiment.


        """
        theta_est = {}
        for key, _ in self.posterior.items():
            if self.likelihood == "Binomial":
                theta_est[key] = random.beta(
                    size=1,
                    a=self.posterior[key]["alpha"],
                    b=self.posterior[key]["beta"],
                )[0]
            elif self.likelihood == "NegativeBinomial":
                theta_est[key] = random.beta(
                    size=1,
                    a=self.posterior[key]["alpha"],
                    b=self.posterior[key]["beta"] * self.posterior[key]["r"],
                )[0]
            elif self.likelihood in ["Poisson", "Exponential"]:
                theta_est[key] = random.gamma(
                    size=1,
                    shape=self.posterior[key]["shape"],
                    scale=1 / self.posterior[key]["rate"],
                )[0]

        return max(theta_est.items(), key=operator.itemgetter(1))[0]

    def update_posterior(self, outcomes: list):
        """
        Takes in a list of dictionaries with the results and updates the Posterior
        distribution for the label.

        outcomes = [{"label": "A", "reward": 1}, {"label":"B", "reward":0}]

        """

        for result in outcomes:
            if self.likelihood == "Poisson":
                self.posterior[result["label"]]["shape"] += result["reward"]
                self.posterior[result["label"]]["rate"] += 1
                self.posterior[result["label"]]["n"] += 1
            if self.likelihood == "Exponential":
                self.posterior[result["label"]]["shape"] += 1
                self.posterior[result["label"]]["rate"] += result["reward"]
                self.posterior[result["label"]]["n"] += 1
            elif self.likelihood == "Binomial":
                self.posterior[result["label"]]["alpha"] += result["reward"]
                self.posterior[result["label"]]["beta"] += 1 - result["reward"]
                self.posterior[result["label"]]["n"] += 1
            elif self.likelihood == "NegativeBinomial":
                self.posterior[result["label"]]["alpha"] += result["reward"]
                self.posterior[result["label"]]["beta"] += 1
                self.posterior[result["label"]]["n"] += 1
        return self.posterior

    def view_posterior_distribution(self, n):
        """
        Creates a visual for the distributions of the different groups in the experiment
        using the current posterior distribution and MonteCarlo simulation.

        Parameters
        ----------
        n:
            The number of samples to take for each group to build the visual

        """
        if self.likelihood == "Poisson":
            plot_values = {
                key: random.gamma(
                    shape=self.posterior[key]["shape"],
                    scale=1 / self.posterior[key]["rate"],
                    size=n,
                )
                for key, _ in self.posterior.items()
            }
        elif self.likelihood == "Exponential":
            plot_values = {
                key: random.gamma(
                    shape=self.posterior[key]["shape"],
                    scale=1 / self.posterior[key]["rate"],
                    size=n,
                )
                for key, _ in self.posterior.items()
            }
        elif self.likelihood == "Binomial":
            plot_values = {
                key: random.beta(
                    a=self.posterior[key]["alpha"],
                    b=self.posterior[key]["beta"],
                    size=n,
                )
                for key, _ in self.posterior.items()
            }
        elif self.likelihood == "NegativeBinomial":
            plot_values = {
                key: random.beta(
                    a=self.posterior[key]["alpha"],
                    b=self.posterior[key]["beta"],
                    size=n,
                )
                for key, _ in self.posterior.items()
            }

        for key, sim_array in plot_values.items():
            sns.distplot(sim_array, hist=False, label=key)
        plt.title("Posterior Distributions")
        plt.legend()
        plt.xlabel("Parameter Value")

    def get_ppd(self, n, label):
        """
        Returns the mean and 95% confidence interval for the posterior predictive
        distribution P(x*|x), which is the distribution of unobserved x values given the observed
        values, P(theta|x) is the posterior distribution.

        """
        if self.likelihood == "Poisson":
            pred_outcome = [
                float(
                    random.poisson(
                        lam=random.gamma(
                            shape=self.posterior[label]["shape"],
                            scale=1 / self.posterior[label]["rate"],
                            size=1,
                        ),
                        size=1,
                    )
                )
                for _ in range(n)
            ]
            summary_stats = {
                "2.5%": percentile(pred_outcome, 2.5),
                "97.5%": percentile(pred_outcome, 97.5),
                "mean": mean(pred_outcome),
            }
        if self.likelihood == "Exponential":
            pred_outcome = [
                float(
                    random.exponential(
                        scale=1
                        / random.gamma(
                            shape=self.posterior[label]["shape"],
                            scale=1 / self.posterior[label]["rate"],
                            size=1,
                        ),
                        size=1,
                    )
                )
                for _ in range(n)
            ]
            summary_stats = {
                "2.5%": percentile(pred_outcome, 2.5),
                "97.5%": percentile(pred_outcome, 97.5),
                "mean": mean(pred_outcome),
            }
        elif self.likelihood == "Binomial":
            pred_outcome = [
                int(
                    random.binomial(
                        n=1,
                        p=random.beta(
                            a=self.posterior[label]["alpha"],
                            b=self.posterior[label]["beta"],
                            size=1,
                        ),
                        size=1,
                    )
                )
                for _ in range(n)
            ]
            summary_stats = {
                "Count - Success": sum(pred_outcome),
                "Count - Fail": len(pred_outcome) - sum(pred_outcome),
                "mean": mean(pred_outcome),
            }
        elif self.likelihood == "NegativeBinomial":
            pred_outcome = [
                int(negative_binomial_sim(self.posterior, label)) for _ in range(n)
            ]
            summary_stats = {
                "2.5%": percentile(pred_outcome, 2.5),
                "97.5%": percentile(pred_outcome, 97.5),
                "mean": mean(pred_outcome),
            }

        return summary_stats

