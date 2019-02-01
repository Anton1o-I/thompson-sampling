from typing import List
from numpy.random import binomial
from numpy import mean
import operator
from base import BaseThompsonSampling


class BernoulliExperiment(BaseThompsonSampling):
    def __init__(self, arms: int = None, priors: List[dict] = None):
        super().__init__()
        default = {"a": 1, "b": 1}
        self._posterior = "beta"
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
            self.posteriors[result["label"]]["a"] += result["reward"]
            self.posteriors[result["label"]]["b"] += 1 - result["reward"]
        return self

    def get_ppd(self, size, label):
        """
        Simulates the posterior predictive distribution for a given
        label and returns the mean, and 95% credible interval.
        """
        pred_outcome = [
            int(
                binomial(
                    n=1,
                    p=self._avail_posteriors[self._posterior](
                        size=1, **self.posteriors[label]
                    ),
                    size=1,
                )
            )
            for _ in range(size)
        ]
        summary_stats = {
            "Count - Success": sum(pred_outcome),
            "Count - Fail": len(pred_outcome) - sum(pred_outcome),
            "mean": mean(pred_outcome),
        }
        return summary_stats
