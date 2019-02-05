
# TODO build out functionality to add priors
class Prior:
    def __init__(self, distribution):
        self.distribution = distribution

    def _gamma_prior(self, mean, interval):
        pass

    def _exponential_prior(self, mean, inteval):
        pass

    def add_prior(self, label: str, mean: int = None, interval: int = None):
        pass

