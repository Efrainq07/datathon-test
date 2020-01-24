import networkx as nx
import matplotlib.pyplot as plt
import operator
import keras
import numpy as np
import metric as met


G = nx.Graph()
G.add_edges_from([('A','B'),('A','D'),('A','C'),('B','F'),('C','G'),('D','G'),('G','H')])

class InputTensor(object):

    def __init__(self,metrics=[]):
        self.metrics = metrics

    def format_function(self,function, netx=False):
        if netx:
            return lambda G: [i[2] for i in function(G)]
        else:
            return lambda G: [function(G,e[0],e[1]) for e in nx.non_edges(G)]

    def evaluate(self,graph=None):
        arr = []
        if graph:
            for func in self.metrics:
                arr.append(func(graph))
        return np.array(arr).transpose()


tn = InputTensor()
tn.metrics=[tn.format_function(met.vecinos_comunes),
            tn.format_function(nx.jaccard_coefficient,netx=True) ]

print(tn.evaluate(G))


'''
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])'''