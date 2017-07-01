# Iris Classification with Fixed-Topology Neuroevolution

An implementation of neuroevolution applied to the classic problem of [Iris flower classification](http://archive.ics.uci.edu/ml/datasets/Iris).

## Methodology 

The network topology used is a Multilayer Perceptron with one hidden layer with 5 neurons.

A population of 800 such networks is initialized with random weights, and successive generations are generated through selection and reproduction. This is repeated for 100 generations, and the "fittest" network in the final generation is used. Fitness is defined as the additive inverse of cross-entropy loss in this implementation.

**Note: hyperparameters are chosen somewhat arbitrarily and have not been tuned.**

### Selection

[Tournament selection](https://en.wikipedia.org/wiki/Tournament_selection) is used to stochastically select networks to be used as "parents" for the next generation. Each tournament consists of 10 networks randomly chosen from the population. The "fittest" of these networks is selected. This process is repeated until 200 networks have been selected.

### Reproduction

The parent networks are randomly paired for reproduction. Each pair produces one child. The process of pairing and reproduction is repeated until there are 800 networks for the next generation.

#### Crossover

Given two parent networks, crossover swaps a randomly chosen neuron between the parents, resulting in two child networks. The fitter of these children is mutated and included in the next generation.

#### Mutation

After crossover, there is a 10% chance that a random weight child network will be mutated.

Types of mutations include

- scaling the weight by a factor in the range (0.5, 1.5)
- adding a random adjustment in the range (-1, 1) to the weight
- inverting the weight by multiplying its value by -1

## Results

Average training accuracy converges to 98.9% by the last generation; the maximum training accuracy reached is 100%.

However, the accuracy is only 57.7% on test data, indicating really bad overfitting. This makes sense since dropout was not used, for the sake of simplicity. 
