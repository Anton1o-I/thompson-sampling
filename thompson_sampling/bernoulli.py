from numpy.random import binomial
from numpy import mean
import operator
from thompson_sampling.base import BaseThompsonSampling
from thompson_sampling.priors import BetaPrior
from typing import List


class BernoulliExperiment(BaseThompsonSampling):
        _default = {"a": 1, "b": 1}
        _posterior = "beta"
        
    def __init__(self, arms: int = None, priors: BetaPrior = None, labels: list = None):

        super().__init__(arms, priors, labels)

    def add_rewards(self, outcomes: List[dict]):
        """
        Takes in a list of dictionaries with the results and updates the Posterior
        distribution for the label.

        outcomes = [{"label": "A", "reward": 1}, {"label":"B", "reward":0}]

        """

        for result in outcomes:
            self.posteriors[result["label"]]["a"] += result["reward"]
            self.posteriors[result["label"]]["b"] += 1 - result["reward"]
        return self

    def get_ppd(self, size) -> List[dict]:
        """
        Simulates the posterior predictive distribution for all available
        posterior distributions and provides Percentage Success & Percentage Failure.
        """
        ppd_stats = []
        for k, _ in self.posteriors.items():
            pred_outcome = [
                int(
                    binomial(
                        n=1,
                        p=self._avail_posteriors[self._posterior](
                            size=1, **self.posteriors[k]
                        ),
                        size=1,
                    )
                )
                for _ in range(size)
            ]
            summary_stats = {
                "Label": k,
                "Percentage - Success": sum(pred_outcome) / size,
                "Percentage - Fail": (len(pred_outcome) - sum(pred_outcome)) / size,
            }
            ppd_stats.append(summary_stats)
        return ppd_stats
