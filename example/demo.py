import os
import networkx as nx
import numpy as np
import dionysus as d
from dgms.format import dgms2diags, dgm2diag, diag2dgm, res2dgms, diags2dgms
from dgms.format import diag_check, flip_dgm, export
from dgms.stat import print_dgm
from dgms.fil import fil_stradegy, graph2dgm

def read_graph(file):
    dir = './data/'
    # file = 'EntoPeduncular_5NN_RANS_ALL_5000.graphml'
    g = nx.read_graphml(dir + file)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g

if __name__=='__main__':
    # file = 'EntoPeduncular_5NN_RANS_ALL_5000.graphml'
    dir = './RNA/'
    file = 'EntoPeduncular_SPAR_RANS_ALL_5000.graphml'
    g = read_graph(file)
    print(nx.degree_histogram(g))
    print(nx.info(g))
    # g = nx.random_geometric_graph(20, 0.5) # create a random graph
    assert nx.is_connected(g)


    # Example 1: node feat example.
    nodefeat = np.random.random((len(g), 1)) # specify node filtration function. Here I am using random filtration function. TODO
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat=nodefeat) # sub-level filtration
    g = fil.build_fv() # compute function value for nodes and edges
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', zigzag=False) # compute the diagram 
    export(diagram, dir=dir, filename='node_fil_dgm.csv')


    # Example 2: edge feat example. Using edge_probability as edge filtration function.
    fil = fil_stradegy(g, fil='edge')
    # fil.edgefeat(func='edge_prob', edge_fil='sup') # use edge probability as an example. Slow for large graph
    fil.edgefeat(func='others', edge_fil='sub', edgefeat=np.random.random((len(g), len(g)))) # edgefeat is a random matrix of (n, n)

    g = fil.build_fv()
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', zigzag=False)
    export(diagram, dir=dir, filename='edge_fil_dgm.csv')


    # Example 3: combined fil
    fil = fil_stradegy(g, fil='combined')

    # set node feat and edge feat value
    fil.nodefeat( nodefeat=np.random.random((len(g), 1)))
    fil.edgefeat( edge_fil='sup', edgefeat=np.random.random((len(g), len(g))))

    g1 = fil.build_fv()
    x = graph2dgm(g1, nodefil='sub', edgefil='sup') # TODO change accordingly
    diagram = x.get_diagram(g1, key='fv', zigzag=True) # set zigzag as True
    export(diagram, dir=dir, filename='combined_dgm.csv')

    exit()
    
    
    # sublevel filtration. fv stands for function value
    for n in g.nodes():
        g.node[n]['fv'] = nx.closeness_centrality(g)
    for u, v in g.edges():
        g[u][v]['fv'] = max(g.node[u]['fv'], g.node[v]['fv'])

    # from filtration on graph nodes to diagram
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag = 'True', parallel_flag = False, zigzag = False)
    for p in diagram:
        print(p)