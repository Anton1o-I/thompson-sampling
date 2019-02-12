import pytest
from thompson_sampling.exponential import ExponentialExperiment
from thompson_sampling.priors import GammaPrior
from pandas import Series


class TestExponentialExperiment:
    def test_init_arms(self):
        exper = ExponentialExperiment(3)
        assert len(exper.posteriors) == 3
        assert isinstance(exper.posteriors, dict)
        for k, _ in exper.posteriors.items():
            assert isinstance(exper.posteriors[k], dict)
            assert exper.posteriors[k] == {"shape": 0.001, "scale": 1000}

    def test_init_custom(self):
        prior = GammaPrior()
        means = Series([0.100, 0.200])
        variances = Series([0.10, None])
        effective_sizes = Series([None, 20])
        labels = Series(["option1", "option2"])
        prior.add_multiple(means, variances, effective_sizes, labels)
        exper = ExponentialExperiment(priors=prior)
        assert len(exper.posteriors) == 2

    def test_update_posterior(self):
        exper = ExponentialExperiment(3)
        exper.add_rewards(
            [{"label": "option1", "reward": 1}, {"label": "option2", "reward": 0}]
        )
        assert exper.posteriors == {
            "option1": {"shape": 1.001, "scale": 0.999},
            "option2": {"shape": 1.001, "scale": 1000.0},
            "option3": {"shape": 0.001, "scale": 1000},
        }

    def test_pull_arm(self):
        exper = ExponentialExperiment(3)
        assert exper.choose_arm() in [key for key, _ in exper.posteriors.items()]

    def test_get_ppd(self):
        exper = ExponentialExperiment(3)
        assert isinstance(exper.get_ppd(size=10000), list)
        assert len(exper.get_ppd(size=1000)) == len(exper.posteriors)
