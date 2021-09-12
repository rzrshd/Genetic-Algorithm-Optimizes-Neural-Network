import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Gaussian distribution, mean=0 and std=1
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # Gaussian distribution, mean=0 and std=1
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers - 2)])

    def feedforward(self, h):
        for b, w in zip(self.biases, self.weights):
            h = self.sigmoid(np.dot(w, h) + b)

        return h

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def loss_function(self, X, y):
        loss = 0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1, 1))
            actual = y[i].reshape(-1, 1)
            loss += np.sum(np.power(predicted - actual, 2) / 2)  # Mean-Squared Error

        return loss

    def accuracy(self, X, y):
        correctly_predicted = 0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1, 1))
            correctly_predicted += int(np.argmax(predicted) == np.argmax(y[i]))

        return correctly_predicted / X.shape[0] * 100


class Genetic_Algorithm_NN:
    def __init__(self, population_size, network_size, mutation_rate, crossover_rate, elite_size, X, y):
        self.population_size = population_size
        self.network_size = network_size
        self.networks = [Network(self.network_size) for i in range(self.population_size)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.X = X[:]
        self.y = y[:]

    def get_random_neuron(self, type):
        network = self.networks[0]
        layer_index, neuron_index = random.randint(0, network.num_layers - 2), 0
        if type == 'weight':
            row = random.randint(0, network.weights[layer_index].shape[0] - 1)
            col = random.randint(0, network.weights[layer_index].shape[1] - 1)
            neuron_index = (row, col)
        elif type == 'bias':
            neuron_index = random.randint(0, network.biases[layer_index].size - 1)

        return (layer_index, neuron_index)

    def get_all_losses(self):
        return [network.loss_function(self.X, self.y) for network in self.networks]

    def get_all_accuracies(self):
        return [network.accuracy(self.X, self.y) for network in self.networks]

    def crossover(self, father, mother):
        network = copy.deepcopy(father)

        # Crossover bias:
        for _ in range(self.networks[0].bias_nitem):
            layer, neuron = self.get_random_neuron('bias')
            if random.uniform(0, 1) < self.crossover_rate:
                network.biases[layer][neuron] = mother.biases[layer][neuron]

        # Crossover weight:
        for _ in range(self.networks[0].weight_nitem):
            layer, neuron = self.get_random_neuron('weight')
            if random.uniform(0, 1) < self.crossover_rate:
                network.weights[layer][neuron] = mother.weights[layer][neuron]

        return network

    def mutation(self, child):
        network = copy.deepcopy(child)

        # Mutate bias:
        for _ in range(self.networks[0].bias_nitem):
            layer, neuron = self.get_random_neuron('bias')
            if random.uniform(0, 1) < self.mutation_rate:
                network.biases[layer][neuron] += random.uniform(-0.5, 0.5)

        # Mutate weight:
        for _ in range(self.networks[0].weight_nitem):
            layer, neuron = self.get_random_neuron('weight')
            if random.uniform(0, 1) < self.mutation_rate:
                network.weights[layer][neuron[0], neuron[1]] += random.uniform(-0.5, 0.5)

        return network

    def genetic_algorithm(self):
        # Calculate losses for each population of neural nework:
        losses_list = list(zip(self.networks, self.get_all_losses()))

        # Sort the network using its loss:
        losses_list.sort(key=lambda x: x[1])

        # Remove first loss as it is not needed anymore:
        losses_list = [loss[0] for loss in losses_list]

        # Keep only the best ones:
        elite_size_index = int(self.population_size * self.elite_size)
        losses_list_top = losses_list[:elite_size_index]

        # Return some non-best ones:
        non_best_index = int((self.population_size - elite_size_index) * self.elite_size)
        for _ in range(random.randint(0, non_best_index)):
            losses_list_top.append(random.choice(losses_list[elite_size_index:]))

        while len(losses_list_top) < self.population_size:
            father = random.choice(losses_list_top)
            mother = random.choice(losses_list_top)
            if father != mother:
                offspring = self.crossover(father, mother)
                offspring = self.mutation(offspring)
                losses_list_top.append(offspring)

        # Next genetation:
        self.networks = losses_list_top

def main():
    df = pd.read_csv('iris.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    population_size = 30 # 30 networks.
    network_size = [4, 6, 5, 3] # Neurons == Input: 4; Hidden1: 6; Hidden2: 5; Output: 3
    mutation_rate = 0.2
    crossover_rate = 0.4
    elite_size = 0.4
    genetic_algorithm_network = Genetic_Algorithm_NN(population_size, network_size, mutation_rate, crossover_rate, elite_size, X, y)
    start_time = time.time()
    for i in range(1000):
        if i % 10 == 0:
            print(f'Current generation is: {i + 1}')
            print('Time taken thus far is: %.1f seconds' % (time.time() - start_time))
            print('Current top network accuracy: %.2f%%\n' % genetic_algorithm_network.get_all_accuracies()[0])

        genetic_algorithm_network.genetic_algorithm()

if __name__ == '__main__':
    main()
