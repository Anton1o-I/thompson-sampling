# thompson-sampling
Thompson Sampling Multi-Armed Bandit for Python

This project is an implementation of a Thompson Sampling approach to a Multi-Armed Bandit. The goal of this project is to easily create and maintain Thompson Sampling experiments.

Currently this project supports experiments where the response follows a Bernoulli or Poisson distribution. Further work will be done to allow for experiments that follow other distributions, with recommendations/collaboration welcome.

## Usage

### Setting up the experiment:
The following method will instantiate the experiment with default priors.
```python
from thompson_sampling.bernoulli import BernoulliExperiment

experiment = BernoulliExperiment(arms=2)
```

If you want set your own priors using the Priors module:
```python

from thompson_sampling.bernoulli import BernoulliExperiment
from thompson_sampling.priors import BetaPrior

pr = BetaPrior()
pr.add_one(mean=0.5, variance=0.2, effective_size=10, label="option1")
pr.add_one(mean=0.6, variance=0.3, effective_size=30, label="option2")
experiment = BernoulliExperiment(priors=pr)
```

### Getting an action:
Randomly chooses which arm to "pull" in the multi-armed bandit:
```python
experiment.choose_arm()
```

### Updating reward:
Updating the information about the different arms by adding reward information:

```python
rewards = [{"label":"option1", "reward":1}, {"label":"option2", "reward":0}]
experiment.add_rewards(rewards)
```

## Installation

### Pip 
```
pip install thompson-sampling
```
