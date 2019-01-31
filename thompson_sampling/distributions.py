from typing import List
from ts import ThompsonSampling


class BernoulliExperiment(ThompsonSampling):
    def __init__(self, arms: int = None, priors: dict = None):
        default = {"alpha": 1, "beta": 1, "n": 0}
        if arms is None and priors is None:
            raise ValueError("Must have either arms or priors specified")
        if arms:
            self.posterior = {f"option_{i+1}": default for i in range(arms)}

        elif priors:
            self.posterior = {f"option_{i+1}": items for items in priors}

    def get_action(self):

