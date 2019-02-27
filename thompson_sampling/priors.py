from typing import List
from thompson_sampling.base import BasePrior
from pandas import isna

# TODO build out functionality to add priors


class BetaPrior(BasePrior):
    def __init__(self):
        """
        Initializes a prior distribution object
        """
        super().__init__()

    def _param_calculator(self, mean: float, variance: float, effective_size: int):
        """
        Hidden method that creates the beta prior given specifications
        """
        if mean >= 1 or mean <= 0:
            raise ValueError(f"mean:{mean} must be in (0,1)")
        if variance <= 0 or variance >= 0.5 ** 2 or variance >= (mean * (1 - mean)):
            raise ValueError(
                f"variance: {variance} must be in (0,{round(min([0.25, mean*(1-mean)]), 3)})"
            )
        if effective_size <= 0:
            raise ValueError(f"effective_size: {effective_size} must be greater then 0")
        alpha = round((((1 - mean) / variance) - (1 / mean)) * (mean ** 2), 3)
        beta = round(alpha * (1 / mean - 1), 3)
        ratio = effective_size / (alpha + beta)  # effective_size = beta+alpha
        return {"a": round(alpha * ratio), "b": round(beta * ratio)}


class GammaPrior(BasePrior):
    def __init__(self):
        super().__init__()

    def _param_calculator(self, mean, variance: None, effective_size: None):
        if not isna(variance):
            if any([mean <= 0, variance <= 0]):
                raise ValueError("Parameters must be positive")
            rate = mean / variance
            shape = mean ** 2 / variance
            scale = 1 / rate
        if not isna(effective_size) and (isna(variance)):
            if any([mean <= 0, effective_size <= 0]):
                raise ValueError("Parameters must be positive")
            rate = effective_size
            shape = mean * effective_size
            scale = 1 / rate
        elif all([isna(variance), isna(effective_size)]):
            raise ValueError("Must specify either variance or effective size")
        return {"shape": round(shape, 3), "scale": round(scale, 3)}
