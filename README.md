# thompson-sampling
Thompson Sampling Experiment in Python

This project is an implementation of a Thompson Sampling approach to a Multi-Armed Bandit. The goal of this project is to easily create and maintain Thompson Sampling experiments.

Currently this project supports experiments where the response follows a Bernoulli or Poisson distribution. Further work will be done to allow for experiments that follow other distributions, with recommendations/collaboration welcome.

## Usage

### Setting up the experiment:
The following method will instantiate the experiment with default priors.
```python
from thompson_sampling.bernoulli import BernoulliExperiment

experiment = BernoulliExperiment(arms=2)
```

If you want set your own priors:
```python

from thompson_sampling.bernoulli import BernoulliExperiment

experiment = BernoulliExperiment(priors=[{"a":10, "b":5}, {"a":1, "b":2}])
```