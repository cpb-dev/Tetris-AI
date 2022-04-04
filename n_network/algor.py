import torch
import torch.nn as nn
import numpy as np

from n_network.boardInfo import get_board_info #because it gets called from tetrisAI, so folder is needed
# import pygame

#NN config
input_size = 8
output_size = 1
min_weights = -1
max_weights = 1

#Population config
elitism = 0.2 #20% elite networks
mutation_prob = 0.2
inherited_weights = 0.5 #50/50 from each parent

device = 'cpu'


class Network(nn.Module): #Class to contain the Network
    #print(nn.Module)
    def __init__(self, output_w=None):
        super(Network, self).__init__()
        if not output_w:
            self.output = nn.Linear(
                input_size, output_size, bias=False).to(device)
            self.output.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.output.weight,
                                   a = min_weights, b = max_weights)
        else:
            self.output = output_w

    def activate(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            #x = x.view(x.size(0), -1)

            x = self.output(x)
        return x

#Creating the population to fill NN, with children and mutations
class Population:
    def __init__(self, size=20, old_population=None):
        self.size = size
        if old_population is None:
            self.models = [Network() for i in range(size)]
        else:
            #When running children will be made from past gen
            self.old_models = old_population.models
            self.old_fitnesses = old_population.fitnesses
            self.models = []
            self.crossover()
            self.mutate()
        self.fitnesses = np.zeros(self.size)

    def crossover(self):
        print("Crossver")
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in range(self.size)]

        #Sorting the models based on current fitness
        sort_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size * elitism:
                #Finding the top 20% of models
                model_c = self.old_models[sort_indices[i]]
            else:
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                                        
                # Probability that each neuron will come from model A
                prob_neuron_from_a = 0.5

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network()

                for j in range(input_size):
                    #With the probability a neuron will come from model A
                    if np.random.random() > prob_neuron_from_a:
                        model_c.output.weight.data[0][j] = model_b.output.weight.data[0][j]
                    else:
                        model_c.output.weight.data[0][j] = model_a.output.weight.data[0][j]

            self.models.append(model_c)

    def mutate(self):
        print("Mutating")
        for model in self.models:
            #Mutating weights by adding Gaussian noises
            for i in range(input_size):
                if np.random.random() < mutation_prob:
                    with torch.no_grad():
                        noise = torch.randn(1).mul_(inherited_weights).to(device)
                        model.output.weight.data[0][i].add_(noise[0])

#Make a function to get the current model score
def get_score(tetris, model):
    area = (tetris != 47).astype(np.int16)

    try: 
        inputs = get_board_info(area)
    except Exception as e:
        return None
    #print (inputs)

    output = model.activate(np.array(inputs))

    return output

