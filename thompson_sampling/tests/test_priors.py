import pytest
from thompson_sampling.priors import BetaPrior, GammaPrior


class TestBetaPrior:
    def test_add_one(self):
        gen = BetaPrior()
        gen.add_one(mean=0.5, variance=0.2, effective_size=10, label="option1")
        assert len(gen.priors) == 1
        assert isinstance(gen.priors, dict)
        assert isinstance(gen.priors["option1"], dict)
        assert gen.priors == {"option1": {"a": 5, "b": 5}}
