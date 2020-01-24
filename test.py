import networkx as nx
import matplotlib.pyplot as plt
import operator

G = nx.Graph()
G.add_edges_from([('A','B'),('A','D'),('A','C'),('B','F'),('C','G'),('D','G'),('G','H')])

print(nx.average_clustering(G))
print(nx.transitivity(G))
print(nx.minimum_edge_cut(G,'C','G'))

G.nodes['A']['community']=0
G.nodes['C']['community']=0
G.nodes['D']['community']=0
G.nodes['G']['community']=0
G.nodes['H']['community']=1
G.nodes['B']['community']=2
G.nodes['F']['community']=2

L = list(nx.ra_index_soundarajan_hopcroft(G))
L.sort(key=operator.itemgetter(2),reverse=True)
print(L)

nx.draw(G, with_labels=True)
plt.show()
