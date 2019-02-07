from typing import List
from pandas import Series, DataFrame

# TODO build out functionality to add priors


class BasePrior:
    def __init__(self):
        self.priors = []

    def _param_calculator(self, *args):
        return self

    def add_one(
        self, mean: int, variance: int, sample_size: int, label: str
    ) -> List[dict]:
        """
        Allows for individual priors to be specified and added to the priors list
        """

        new_prior = {label: self._param_calculator(mean, variance, sample_size)}
        self.priors.append(new_prior)
        return self

    def add_multiple(
        self, means: Series, variances: Series, sample_sizes: Series, labels: Series
    ) -> List[dict]:
        """
        Allows for a group of priors to be specified at once

        information: DataFrame
        """
        pass


class BetaPrior(BasePrior):
    def __init__(self):
        """
        Initializes a prior distribution object
        """
        super().__init__()

    def _param_calculator(self, mean: float, variance: float, sample_size: int):
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


class GammaPrior(self):
    def __init__(self):
        self.priors = []

    def _param_calculator(self, mean, variance, sample_size):
        shape = variance/mean
        scale = mean**2/variance
        return ({"shape": shape, "scale": scale)


