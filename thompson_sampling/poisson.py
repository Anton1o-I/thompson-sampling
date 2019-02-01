from typing import List
from numpy.random import poisson
from numpy import mean, percentile
import operator
from thompson_sampling.base import BaseThompsonSampling


class PoissonExperiment(BaseThompsonSampling):
    def __init__(self, arms: int = None, priors: List[dict] = None):
        super().__init__()
        default = {"shape": 0.001, "scale": 1000}
        self._posterior = "gamma"
        if arms is None and priors is None:
            raise ValueError("Must have either arms or priors specified")
        if arms:
            self.posteriors = {f"option_{i+1}": default.copy() for i in range(arms)}
        elif priors:
            self.posteriors = {f"option_{i+1}": items.copy() for items in priors}

    def update_posterior(self, outcomes: List[dict]):
        """
        Takes in a list of dictionaries with the results and updates the Posterior
        distribution for the label.

        outcomes = [{"label": "A", "reward": 1}, {"label":"B", "reward":0}]

        """
        for result in outcomes:
            self.posteriors[result["label"]]["shape"] += result["reward"]
            self.posteriors[result["label"]]["scale"] = 1 / (
                1 / self.posteriors[result["label"]]["scale"] + 1
            )
        return self

    def get_ppd(self, size, label):
        """
        Simulates the posterior predictive distribution for a given
        label and returns the mean, and 95% credible interval.
        """
        pred_outcome = [
            int(
                poisson(
                    lam=self._avail_posteriors[self._posterior](
                        size=1, **self.posteriors[label]
                    ),
                    size=1,
                )
            )
            for _ in range(size)
        ]
        summary_stats = {
            "95% Credible Interval": (
                percentile(pred_outcome, 2.5),
                percentile(pred_outcome, 97.5),
            ),
            "mean": mean(pred_outcome),
        }
        return summary_stats
