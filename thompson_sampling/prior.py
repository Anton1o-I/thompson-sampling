from typing import List
from pandas import DataFrame

# TODO build out functionality to add priors
class Prior:
    def __init__(self, distribution: str = None):
        """
        Initializes a prior distribution object
        Specify either "beta" or "gamma" distribution
        """

        if not distribution:
            raise ValueError("Need to specify a distribution, either 'Gamma' or 'Beta'")
        avail_priors = ["gamma", "beta"]
        if distribution.lower() not in avail_priors:
            raise ValueError(
                f"{distribution} not recognized, specify either 'gamma' or 'beta'"
            )
        self.distribution = distribution.lower()
        self.priors = []

    def _gamma_prior(self, mean, variance):
        pass

    def _beta_prior(self, mean: float, variance: float, sample_size: int):
        """
        Hidden method that creates the beta prior given specifications
        """
        if mean >= 1 or mean <= 0:
            raise ValueError(f"mean:{mean} must be in (0,1)")
        if variance <= 0 or variance >= 0.5 ** 2 or variance >= (mean * (1 - mean)):
            raise ValueError(
                f"variance: {variance} must be in (0,{round(min([0.25, mean*(1-mean)]), 3)})"
            )
        if sample_size <= 0:
            raise ValueError(f"sample_size: {sample_size} must be greater then 0")
        alpha = round((((1 - mean) / variance) - (1 / mean)) * (mean ** 2), 3)
        beta = round(alpha * (1 / mean - 1), 3)
        ratio = sample_size / (alpha + beta)  # sample size = beta+alpha
        return {"a": alpha * ratio, "b": beta * ratio}

    def add_one(
        self, mean: int, variance: int, sample_size: int, label: str
    ) -> List[dict]:
        """
        Allows for individual priors to be specified and added to the priors list
        """
        if self.distribution == "beta":
            new_prior = {label: self._beta_prior(mean, variance, sample_size)}
            self.priors.append(new_prior)
        return self

    def add_multiple(self, information: DataFrame) -> List[dict]:
        """
        Allows for a group of priors to be specified at once

        information: DataFrame
        """
        pass
