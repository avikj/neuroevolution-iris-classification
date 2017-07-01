from sklearn import datasets
from sklearn.metrics import log_loss
import numpy as np
import random


def main():
  iris = datasets.load_iris()

  # shuffle inputs and outputs in parallel
  rng_state = np.random.get_state()
  np.random.shuffle(iris.data)
  np.random.set_state(rng_state)
  np.random.shuffle(iris.target)

  # split dataset
  train_X = iris.data[:int(0.7 * len(iris.data))]
  train_X = normalize_features(train_X, train_X)
  train_y = one_hot(iris.target[:int(0.7 * len(iris.data))], 3)

  test_X = iris.data[int(0.7 * len(iris.data)):]
  test_X = normalize_features(test_X, train_X)
  test_y = one_hot(iris.target[int(0.7 * len(iris.data)):], 3)

  evolved_network = evolve(train_X, train_y, [random_network_params() for i in range(800)], 800, 100)
  test_accuracy = accuracy(evolved_network, test_X, test_y)
  print "================================================================================="
  print 'test accuracy:', test_accuracy


##########################################
# Implementation of Neuroevolution Steps #
##########################################
def evolve(X, y, initial_networks, population_size, num_generations):
  current_generation_networks = initial_networks
  for generation in range(num_generations):
    network_fitnesses = [fitness(network_params, X, y) for network_params in current_generation_networks]
    print 'generation:', generation, ', mean train acc:', np.mean([accuracy(network, X, y) for network in current_generation_networks]), ', max train acc:', np.max([accuracy(network, X, y) for network in current_generation_networks])
    selected_network_indices = select(network_fitnesses, population_size / 4, 4)
    survivors = [current_generation_networks[i] for i in selected_network_indices]
    current_generation_networks = next_generation(survivors, X, y)
  return current_generation_networks[np.argmax([accuracy(network, X, y) for network in current_generation_networks])]


# apply pairing, crossover, and mutation to compute next generation of networks
def next_generation(survivors, X, y):
  next_generation_networks = []
  for i in range(8):
    for mommy, daddy in random_pairs(survivors):
      child = crossover_networks(mommy, daddy, X, y)
      next_generation_networks.append(child)
  return next_generation_networks


# return one child, determined by randomly swapping a neuron in parents and choosing more fit result
def crossover_networks(mommy_params, daddy_params, train_X, train_y):
  child_m = {'W_1': np.copy(mommy_params['W_1']), 'b_1': np.copy(mommy_params['b_1']), 'W_2': np.copy(mommy_params['W_2']), 'b_2': np.copy(mommy_params['b_2'])}
  child_d = {'W_1': np.copy(daddy_params['W_1']), 'b_1': np.copy(daddy_params['b_1']), 'W_2': np.copy(daddy_params['W_2']), 'b_2': np.copy(daddy_params['b_2'])}
  rand = np.random.rand()
  # randomly select layer to perform crossover on
  if rand < 0.7:  # crossover hidden layer
    child_m['W_1'], child_d['W_1'] = crossover_layer_weights(mommy_params['W_1'], daddy_params['W_1'])
  else:  # crossover output layer
    child_m['W_2'], child_d['W_2'] = crossover_layer_weights(mommy_params['W_2'], daddy_params['W_2'])

  if fitness(child_d, train_X, train_y) > fitness(child_m, train_X, train_y):
    return mutate_network(child_d)
  else:
    return mutate_network(child_m)


# return two children layers from randomly swapping a neuron/weight in parent layers
def crossover_layer_weights(mommy_W, daddy_W, crossover_type="neuron"):
  if crossover_type == 'neuron':
    # swap weights for random neuron in layer (matrix column)
    rand_column = np.random.randint(mommy_W.shape[1])
    mommy_column = np.copy(mommy_W[:, rand_column])
    daddy_column = np.copy(daddy_W[:, rand_column])
    child_m_W = np.copy(mommy_W)
    child_d_W = np.copy(daddy_W)
    child_m_W[:, rand_column] = daddy_column
    child_d_W[:, rand_column] = mommy_column
    return (child_m_W, child_d_W)


# randomly mutate network weights in each layer
def mutate_network(network_params):
  network_params['W_1'] = mutate_weights(network_params['W_1'])
  network_params['W_2'] = mutate_weights(network_params['W_2'])
  return network_params


# randomly mutate weights in a layer
def mutate_weights(W):
  if np.random.rand() < 0.1:
    rand = np.random.rand()
    i = np.random.randint(W.shape[0])
    j = np.random.randint(W.shape[1])
    if rand < 0.5:
      W[i][j] *= np.random.uniform(0.5, 1.5)
    elif rand < 0.9:
      W[i][j] += np.random.uniform(-1, 1)
    else:
      W[i][j] *= -1
  return W


# select n networks to be used as parents in the next generation, using tournament selection algorithm
def select(network_fitnesses, n, k=10):
  selected_network_indices = []
  for i in range(n):
    competitors_indices = np.random.choice(len(network_fitnesses), k, replace=False)
    winner_index = competitors_indices[np.argmax(np.array(network_fitnesses)[competitors_indices])]
    selected_network_indices.append(winner_index)
  return np.array(selected_network_indices)


# compute fitness of network as additive inverse of loss
def fitness(network_params, X, y):
  return -log_loss(y, predict_proba(X, network_params))


# randomly pair networks in list of tuples
def random_pairs(networks):
  random.shuffle(networks)
  result = []
  for i in range(0, len(networks), 2):
    result.append((networks[i], networks[i + 1]))
  return result


#########################################
# Neural Network Definition and Metrics #
#########################################
def predict_proba(X, network_params):
  h = sigmoid(np.matmul(X, network_params['W_1']) + network_params['b_1'])
  y = softmax(np.matmul(h, network_params['W_2']) + network_params['b_2'])
  return y


def predict(X, network_params):
  r = np.argmax(predict_proba(X, network_params), axis=1)
  return r


def sigmoid(X):
  return 1 / (1 + np.exp(-X))


def softmax(X):
  return np.divide(np.exp(X), np.sum(np.exp(X), axis=1)[:, np.newaxis])


def accuracy(network_params, X, y):
  return np.mean(np.equal(predict(X, network_params), np.argmax(y, axis=1)))


def random_network_params():
  HIDDEN_LAYER_SIZE = 5
  stddev = 0.1
  W_1 = np.random.randn(4, HIDDEN_LAYER_SIZE) * stddev
  b_1 = np.random.randn(HIDDEN_LAYER_SIZE) * stddev
  W_2 = np.random.randn(HIDDEN_LAYER_SIZE, 3) * stddev
  b_2 = np.random.randn(3) * stddev
  return {'W_1': W_1, 'b_1': b_1, 'W_2': W_2, 'b_2': b_2}


#################
# Preprocessing #
#################
def normalize_features(X, train_X):
  return (X - np.mean(train_X, axis=0)) / np.std(train_X, axis=0)


def one_hot(y, num_classes):
  result = np.zeros((len(y), num_classes))
  for i in range(len(y)):
    result[i][y[i]] = 1
  return result


main()
