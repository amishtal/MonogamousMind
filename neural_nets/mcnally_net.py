import array
import random

import numpy as np
import scipy
from scipy.stats import nbinom

from deap import base, creator, tools

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

class ContextNode(object):
    def __init__(self):
        self.W_out = np.random.random()

        self.value = 0.0

    def activate(self, data_in):
        prev_value = self.value
        self.value = data_in

        return prev_value * self.W_out

    def reset(self):
        self.value = 0.0

    def mutate_weights(self, prob):
        if np.random.ran() <= prob:
            self.W_out += 0.5 * np.random.randn()

class HiddenNode(object):
    def __init__(self, n_in, n_out, context_prob=0.0):
        self.W_in = 2.0 * np.random.randn(n_in,1)
        self.bias = random.uniform(-1.0, 1.0)

        self.W_out = 2.0 * np.random.randn(1, n_out)

        if np.random.rand() <= context_prob:
            self.context_node = ContextNode()
        else:
            self.context_node = None

    def activate(self, data_in):
        _sum = np.dot(data_in, self.W_in) + self.bias
        if self.context_node is not None:
            _sum += self.context_node.value

        activation = sigmoid(_sum)
        weighted_activation = np.dot(activation, self.W_out)

        return weighted_activation
        
    def reset(self):
        if self.context_node is not None:
            self.context_node.reset()

    def mutate_weights(self, prob):
        shape = self.W_in.shape
        mask = np.random.rand(*shape) <= prob
        change = 0.5 * np.random.randn(*shape)
        self.W_in += (mask * change)

        if np.random.rand() <= prob:
            self.bias += 0.5 * np.random.randn()

        if not self.has_context_node:
            self.context_node.mutate_weights(prob)

    def add_context_node(self):
        self.context_node = ContextNode()

    def remove_context_node(self):
        self.context_node = None

    def has_context_node(self):
        return self.context_node is not None

class McNallyNet(object):
    def __init__(self, n_in, max_n_hidden, n_out, max_init_hidden):
        self.n_in = n_in
        self.n_out = n_out
        self.max_n_hidden = max_n_hidden

        if max_init_hidden is None:
            max_init_hidden = self.max_n_hidden

        n_init_hidden = random.randint(0, max_init_hidden)

        self.nodes = []
        for i in range(n_init_hidden):
            self.nodes.append(HiddenNode(self.n_in, self.n_out))

        self.output_bias = 2 * np.random.random((1,self.n_out)) - 1

        self.initial_move = np.random.randint(0, 2)

    def activate(self, data_in):
        out = None
        if len(self.nodes) > 0:
            out = np.zeros((1, self.n_out))
            for node in self.nodes:
                out += node.activate(data_in)
            out += self.output_bias
            out = sigmoid(out)
        else:
            out = self.initial_move

        return out

    def reset(self):
        for node in self.nodes:
            node.reset()

    def count_cognitive_nodes(self):
        return len(self.nodes)

    def count_context_nodes(self):
        n = 0
        for node in self.nodes:
            if node.context_node is not None:
                n += 1

        return n

    def get_intelligence(self):
        return self.count_context_nodes() + self.count_cognitive_nodes()

    def mutate_weights(self, prob):
        if np.random.rand() <= prob:
            self.initial_move = 1 - self.initial_move
        for node in self.nodes:
            node.mutate_weights(prob)

    def remove_cognitive_node(self):
        i = np.random.choice(range(0, len(self.nodes)))
        del self.nodes[i]

    def add_cognitive_node(self):
        self.nodes.append(HiddenNode(self.n_in, self.n_out))

    def remove_context_node(self):
        has_context_nodes = [node.has_context_node() for node in self.nodes]
        valid = filter(lambda x: x[0], zip(has_context_nodes, self.nodes))
        choice = np.random.choice([x[1] for x in valid])
        choice.remove_context_node()

    def add_context_node(self):
        has_context_nodes = [node.has_context_node() for node in self.nodes]
        #print '{}'.format(sum(has_context_nodes))
        valid = filter(lambda x: not x[0], zip(has_context_nodes, self.nodes))
        choice = np.random.choice([x[1] for x in valid])
        choice.add_context_node()

    def mutate_structure(self, prob):
        if np.random.rand() <= prob:
            #print 'Performing structural mutation!'
            # Determine which structural mutations are possible given
            # the network's current structure.
            n_cognitive_nodes = self.count_cognitive_nodes()
            n_context_nodes = self.count_context_nodes()
            possible = []
            if n_cognitive_nodes > 0:
                possible.append(StructuralMutation.RemoveCognitiveNode)
            if n_cognitive_nodes < self.max_n_hidden:
                possible.append(StructuralMutation.AddCognitiveNode)

            if n_context_nodes > 0:
                possible.append(StructuralMutation.RemoveContextNode)
            if n_context_nodes < n_cognitive_nodes:
                possible.append(StructuralMutation.AddContextNode)

            #print 'possible: {}'.format(possible)
            choice = np.random.choice(possible)
            #print 'choice: {}'.format(choice)

            if choice == StructuralMutation.RemoveCognitiveNode:
                #print 'Removing a cognitive node'
                self.remove_cognitive_node()
            elif choice == StructuralMutation.AddCognitiveNode:
                #print 'Adding a cognitive node'
                self.add_cognitive_node()
            elif choice == StructuralMutation.RemoveContextNode:
                #print 'Removing a context node'
                self.remove_context_node()
            elif choice == StructuralMutation.AddContextNode:
                #print 'Adding a context node'
                #print '# nodes: {}'.format(len(self.nodes))
                self.add_context_node()

def mutate(net, prob_mutate_weights, prob_mutate_structure):
    net.mutate_weights(prob_mutate_weights)
    net.mutate_structure(prob_mutate_structure)

    return net

def crossover(net1, net2):
    net1_new_nodes = []
    net2_new_nodes = []

    random.shuffle(net1.nodes)
    random.shuffle(net2.nodes)

    m1 = len(net1.nodes) / 2
    m2 = len(net2.nodes) / 2

    net1_new_nodes = net1.nodes[0:m1] + net2.nodes[m2:]
    net2_new_nodes = net2.nodes[0:m2] + net1.nodes[m1:]

    net1.nodes = net1_new_nodes
    net2.nodes = net2_new_nodes

    return net1, net2
