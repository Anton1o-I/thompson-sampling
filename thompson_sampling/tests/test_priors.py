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
        assert isinstance(gen.priors, dict)
        assert isinstance(gen.priors["option1"], dict)
        assert gen.priors == {
            "option1": {"a": 2.0, "b": 8.0},
            "option2": {"a": 5.0, "b": 5.0},
        }

