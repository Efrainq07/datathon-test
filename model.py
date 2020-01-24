import networkx as nx
import matplotlib.pyplot as plt
import operator
import keras
import numpy as np
import metric as met
import random

from keras.models import Sequential
from keras.layers import Dense,Activation


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
            tn.format_function(nx.jaccard_coefficient,netx=True),
            tn.format_function(nx.resource_allocation_index,netx=True),
            tn.format_function(nx.adamic_adar_index,netx=True),
            tn.format_function(nx.preferential_attachment,netx=True)]

X_train = tn.evaluate(G)

Y_train = np.array([random.randint(0,2) for i in range(len(X_train))])

model = Sequential([
    Dense(32, input_shape=(len(tn.metrics),),activation='relu'),
    Dense(10),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['binary_accuracy'])

model.fit(X_train, Y_train, 
          batch_size=len(X_train), epochs=10, verbose=1)