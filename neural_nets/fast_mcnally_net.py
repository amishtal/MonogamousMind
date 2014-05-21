import array
import random

import numpy as np
import scipy
from scipy.stats import nbinom

from deap import base, creator, tools

import neural_nets
import neural_nets.mcnally_net

'''
A class of evolvable artificial neural networks with
the following features:

  - Layered architecture
  - Each neuron can have a single associated 'context' neuron
  - Each layer has a maximum size

'''

'''
Should make a general 'HiddenNode' based neural network class
that consists of modules (a hidden node and its associated weights,
or perhaps more complicated paths from input to output). The
parameters of modules would be targets for mutation, while crossover
would mix and match modules.
'''


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class StructuralMutation(object):
    AddCognitiveNode = 0
    RemoveCognitiveNode = 1
    AddContextNode = 2
    RemoveContextNode = 3

# Implementation of neural network architecture used in McNally's paper
# that uses matrix multiplications for increased efficiency.
class McNallyNet(neural_nets.mcnally_net.McNallyNet):
    def __init__(self, n_in, max_n_hidden, n_out, max_init_hidden):
        self.n_in = n_in
        self.n_out = n_out
        self.max_n_hidden = max_n_hidden

        if max_init_hidden is None:
            max_init_hidden = self.max_n_hidden

        n_init_hidden = random.randint(0, max_init_hidden)
        self.n_cognitive_nodes = n_init_hidden
        self.n_context_nodes = 0 

        self.W_h = np.random.randn(n_in, n_init_hidden)
        self.b_h = np.random.randn(1, n_init_hidden)

        self.W_recurr = np.zeros((1,n_init_hidden))

        self.prev_hidden = np.zeros((1, n_init_hidden))

        self.W_out = np.random.randn(n_init_hidden, n_out)
        self.b_out = np.random.randn(1, n_out)

        self.initial_move = np.random.randint(0, 2)

    def activate(self, data_in):
        if self.count_cognitive_nodes() == 0:
            return self.initial_move
        else:
            #print ''
            #print 'n_cognitive_nodes: {}'.format(self.n_cognitive_nodes)
            #print 'n_context_nodes: {}'.format(self.n_context_nodes)
            #print 'shape of `data_in`: {}'.format(data_in.shape)
            #print 'shape of `W_h`: {}'.format(self.W_h.shape)
            #print 'shape of `b_h`: {}'.format(self.b_h.shape)
            #print 'shape of `W_out`: {}'.format(self.W_out.shape)
            #print 'shape of `b_out`: {}'.format(self.b_out.shape)
            hidden_sum = np.dot(data_in, self.W_h) + self.b_h + \
                    (self.prev_hidden * self.W_recurr)

            hidden = sigmoid(hidden_sum)
            self.prev_hidden = hidden

            #print 'shape of `hidden`: {}'.format(hidden.shape)
            output_sum = np.dot(hidden, self.W_out) + self.b_out
            output = sigmoid(output_sum)

            return output

    def reset(self):
        self.prev_hidden = np.zeros((1, self.n_cognitive_nodes))

    def count_cognitive_nodes(self):
        return self.n_cognitive_nodes
    def count_context_nodes(self):
        return self.n_context_nodes

    def remove_context_node(self):
        if self.n_context_nodes > 0:
            self.n_context_nodes -= 1
            _, possible = np.nonzero(self.W_recurr)
            choice = np.random.choice(possible)

            self.W_recurr[0, choice] = 0.0

    def add_context_node(self):
        if self.n_cognitive_nodes > 0 and self.n_context_nodes < self.n_cognitive_nodes:
            self.n_context_nodes += 1
            possible = np.argwhere(self.W_recurr == 0)
            choice = np.random.choice(possible[:,1])

            self.W_recurr[0, choice] = np.random.randn()
            
    def remove_cognitive_node(self):
        if self.n_cognitive_nodes == 1:
            self.n_cognitive_nodes = 0
            self.n_context_nodes = 0

            self.W_h = np.ndarray((self.n_in,0))
            self.b_h = np.ndarray((1,0))

            self.W_recurr = np.ndarray((1,0))

            self.prev_hidden = np.ndarray((1,0))

            self.W_out = np.ndarray((0,self.n_out))

        elif self.n_cognitive_nodes > 0:
            self.n_cognitive_nodes -= 1
            ind = np.random.randint(0, self.n_cognitive_nodes)

	    self.W_h = np.concatenate((self.W_h[:, 0:ind], self.W_h[:,ind+1:]), axis=1)
	    self.b_h = np.concatenate((self.b_h[:, 0:ind], self.b_h[:,ind+1:]), axis=1)

            if not (self.W_recurr[0, ind] == 0):
                self.n_context_nodes -= 1
	    self.W_recurr = np.concatenate((self.W_recurr[:, 0:ind], self.W_recurr[:,ind+1:]), axis=1)
	    self.prev_hidden = np.concatenate((self.prev_hidden[:, 0:ind], self.prev_hidden[:,ind+1:]), axis=1)

	    self.W_out = np.concatenate((self.W_out[0:ind, :], self.W_out[ind+1:,:]), axis=0)

    def add_cognitive_node(self):
        if self.n_cognitive_nodes == 0:
            self.n_cognitive_nodes += 1

            self.W_h = np.random.randn(self.n_in, self.n_cognitive_nodes)
            self.b_h = np.random.randn(1, self.n_cognitive_nodes)

            self.W_recurr = np.zeros((1,self.n_cognitive_nodes))

            self.prev_hidden = np.zeros((1, self.n_cognitive_nodes))

            self.W_out = np.random.randn(self.n_cognitive_nodes, self.n_out)

        elif self.n_cognitive_nodes < self.max_n_hidden:
            self.n_cognitive_nodes += 1
            self.W_h = np.concatenate((self.W_h, np.random.randn(self.n_in, 1)), axis=1)
            self.b_h = np.concatenate((self.b_h, np.random.randn(1, 1)), axis=1)

            self.W_recurr = np.concatenate((self.W_recurr, np.zeros((1,1))), axis=1)
            self.prev_hidden = np.concatenate((self.prev_hidden, np.zeros((1,1))), axis=1)

            self.W_out = np.concatenate((self.W_out, np.random.randn(1, self.n_out)), axis=0)


    def mutate_weights(self, prob):
        weights = [self.W_h, self.b_h, self.W_out, self.b_out]
        for w in weights:
            shape = w.shape
            w = w + (np.random.rand(*shape) < prob) * 0.5*np.random.randn(*shape)

        shape = self.W_recurr.shape
        valid = np.logical_not(self.W_recurr == 0.0)
        mask = (np.random.rand(*shape) < prob) * valid
        self.W_recurr = self.W_recurr + mask * 0.5*np.random.randn(*shape)

def mutate(net, prob_mutate_weights, prob_mutate_structure):
    net.mutate_weights(prob_mutate_weights)
    net.mutate_structure(prob_mutate_structure)

    return net

def crossover(net1, net2):
    n1 = net1.count_cognitive_nodes()
    m1 = n1/2
    n2 = net2.count_cognitive_nodes()
    m2 = n2/2

    if n1 == 0:
        net1_indices = []
    else:
        net1_indices = np.random.choice(range(n1), size=n1, replace=False)

    if n2 == 0:
        net2_indices = []
    else:
        net2_indices = np.random.choice(range(n2), size=n2, replace=False)

    new_net1_W_h = np.concatenate((np.atleast_2d(net1.W_h[:,net1_indices[0:m1]]), np.atleast_2d(net2.W_h[:,net2_indices[m2:]])), axis=1)
    new_net2_W_h = np.concatenate((net1.W_h[:,net1_indices[m1:]], net2.W_h[:,net2_indices[0:m2]]), axis=1)

    new_net1_W_recurr = np.concatenate((net1.W_recurr[:,net1_indices[0:m1]], net2.W_recurr[:,net2_indices[m2:]]), axis=1)
    new_net2_W_recurr = np.concatenate((net1.W_recurr[:,net1_indices[m1:]], net2.W_recurr[:,net2_indices[0:m2]]), axis=1)

    new_net1_prev_hidden = np.concatenate((net1.prev_hidden[:,net1_indices[0:m1]], net2.prev_hidden[:,net2_indices[m2:]]), axis=1)
    new_net2_prev_hidden = np.concatenate((net1.prev_hidden[:,net1_indices[m1:]], net2.prev_hidden[:,net2_indices[0:m2]]), axis=1)

    new_net1_b_h = np.concatenate((net1.b_h[:,net1_indices[0:m1]], net2.b_h[:,net2_indices[m2:]]), axis=1)
    new_net2_b_h = np.concatenate((net1.b_h[:,net1_indices[m1:]], net2.b_h[:,net2_indices[0:m2]]), axis=1)

    new_net1_W_out = np.concatenate((net1.W_out[net1_indices[0:m1],:], net2.W_out[net2_indices[m2:],:]), axis=0)
    new_net2_W_out = np.concatenate((net1.W_out[net1_indices[m1:],:], net2.W_out[net2_indices[0:m2],:]), axis=0)

    net1.W_h = new_net1_W_h
    net2.W_h = new_net2_W_h

    net1.W_recurr = new_net1_W_recurr
    net2.W_recurr = new_net2_W_recurr

    net1.prev_hidden = new_net1_prev_hidden
    net2.prev_hidden = new_net2_prev_hidden

    net1.b_h = new_net1_b_h
    net2.b_h = new_net2_b_h

    net1.W_out = new_net1_W_out
    net2.W_out = new_net2_W_out

    net1.n_cognitive_nodes = net1.W_h.shape[1]
    net1.n_context_nodes = len(np.nonzero(net1.W_recurr)[0])

    net2.n_cognitive_nodes = net2.W_h.shape[1]
    net2.n_context_nodes = len(np.nonzero(net2.W_recurr)[0])

    return net1, net2
