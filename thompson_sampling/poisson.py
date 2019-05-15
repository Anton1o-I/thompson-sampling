from typing import List
from numpy.random import poisson
from numpy import mean, percentile
import operator
from thompson_sampling.base import BaseThompsonSampling
from thompson_sampling.priors import GammaPrior


class PoissonExperiment(BaseThompsonSampling):
    _default = {"shape": 0.001, "scale": 1000}
    _posterior = "gamma"

    def __init__(
        self, arms: int = None, priors: GammaPrior = None, labels: list = None
    ):

        super().__init__(arms, priors, labels)

    def add_rewards(self, outcomes: List[dict]):
        """
        Takes in a list of dictionaries with the results and updates the Posterior
        distribution for the label.

        outcomes = [{"label": "A", "reward": 1}, {"label":"B", "reward":0}]

        """
        for result in outcomes:
            self.posteriors[result["label"]]["shape"] += result["reward"]
            self.posteriors[result["label"]]["scale"] = round(
                1 / (1 / self.posteriors[result["label"]]["scale"] + 1), 4
            )
        return self

    def get_ppd(self, size):
        """
        Simulates the posterior predictive distribution for a given
        label and returns the mean, and 95% credible interval.
        """
        ppd_stats = []
        for k, _ in self.posteriors.items():
            pred_outcome = [
                int(
                    poisson(
                        lam=self._avail_posteriors[self._posterior](
                            size=1, **self.posteriors[k]
                        ),
                        size=1,
                    )
                )
                for _ in range(size)
            ]

            summary_stats = {
                "Label": k,
                "95% Credible Interval": (
                    round(percentile(pred_outcome, 2.5), 3),
                    round(percentile(pred_outcome, 97.5), 3),
                ),
                "mean": round(mean(pred_outcome), 3),
            }
            ppd_stats.append(summary_stats)
        return ppd_stats
