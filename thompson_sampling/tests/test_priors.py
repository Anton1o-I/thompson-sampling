import pytest
from thompson_sampling.priors import BetaPrior, GammaPrior
from pandas import Series


class TestBetaPrior:
    def test_add_one_success(self):
        gen = BetaPrior()
        gen.add_one(mean=0.5, variance=0.2, effective_size=10, label="option1")
        assert len(gen.priors) == 1
        assert isinstance(gen.priors, dict)
        assert isinstance(gen.priors["option1"], dict)
        assert gen.priors == {"option1": {"a": 5, "b": 5}}

    def test_add_one_value_errors(self):
        gen = BetaPrior()
        mean_errors = [-1, 0, 1.2]
        variance_errors = [-1, 0, 1]
        effective_size_errors = [-1, 0]
        with pytest.raises(ValueError):
            gen.add_one(mean=0.5, variance=0.5, effective_size=10, label="option1")
            for item in mean_errors:
                gen.add_one(mean=item, variance=0.1, effective_size=10, label="option1")
            for item in variance_errors:
                gen.add_one(mean=0.5, variance=item, effective_size=10, label="option1")
            for item in effective_size_errors:
                gen.add_one(
                    mean=0.5, variance=0.2, effective_size=item, label="option1"
                )

    def test_add_multiple(self):
        gen = BetaPrior()
        means = Series([0.2, 0.5])
        variances = Series([0.02, 0.2])
        effective_sizes = Series([10, 10])
        labels = Series(["option1", "option2"])
        gen.add_multiple(means, variances, effective_sizes, labels)
        assert len(gen.priors) == 2
        assert gen.priors == {
            "option1": {"a": 2.0, "b": 8.0},
            "option2": {"a": 5.0, "b": 5.0},
        }

    def test_add_multiple_error(self):
        gen = BetaPrior()
        means = Series([0.5, 0.6])
        variances = Series([0.24, 0.34, None])
        effective_sizes = Series([20, 10, 50])
        labels = Series(["option1", "option2", "option3"])
        with pytest.raises(ValueError):
            gen.add_multiple(means, variances, effective_sizes, labels)


class TestGammaPrior:
    def test_add_one_success(self):
        gen = GammaPrior()
        params = [
            {"mean": 100, "variance": 20, "effective_size": None},
            {"mean": 100, "effective_size": 20, "variance": None},
        ]
        for i, item in enumerate(params):
            gen.add_one(label=f"option{i}", **item)
        assert len(gen.priors) == 2
        assert isinstance(gen.priors, dict)
        assert isinstance(gen.priors["option1"], dict)
        assert gen.priors == {
            "option0": {"shape": 500, "scale": .2},
            "option1": {"shape": 2000, "scale": .05},
        }

    def test_add_one_value_errors(self):
        gen = GammaPrior()
        mean_errors = [-1, 0]
        variance_errors = [-1, 0]
        effective_size_errors = [-1, 0]
        with pytest.raises(ValueError):
            for item in mean_errors:
                gen.add_one(mean=item, variance=10, effective_size=10, label="option1")
            for item in variance_errors:
                gen.add_one(mean=50, variance=item, effective_size=10, label="option1")
            for item in effective_size_errors:
                gen.add_one(
                    mean=50, variance=None, effective_size=item, label="option1"
                )
            gen.add_one(mean=100, variance=None, effective_size=None, label="option1")

    def test_add_multiple(self):
        gen = GammaPrior()
        means = Series([100, 200, 300])
        variances = Series([20, 10, None])
        effective_sizes = Series([None, None, 20])
        labels = Series(["option1", "option2", "option3"])
        gen.add_multiple(means, variances, effective_sizes, labels)
        assert len(gen.priors) == 3
        assert gen.priors == {
            "option1": {"shape": 500.0, "scale": 0.2},
            "option2": {"shape": 4000.0, "scale": 0.05},
            "option3": {"shape": 6000.0, "scale": 0.05},
        }

    def test_add_multiple_error(self):
        gen = GammaPrior()
        means = Series([100, 200])
        variances = Series([20, 10, None])
        effective_sizes = Series([None, None])
        labels = Series(["option1", "option2", "option3"])
        with pytest.raises(ValueError):
            gen.add_multiple(means, variances, effective_sizes, labels)
