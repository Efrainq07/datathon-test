import networkx as nx
import numpy as np

def vecinos_comunes(G,e1,e2):
    return len(list(nx.common_neighbors(G,e1,e2)))