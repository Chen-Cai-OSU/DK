from dgms.format import dgm2diag, diag2dgm
import networkx as nx
from numpy import random
import dionysus as d
import numpy as np
from dgms.test import randomdgms

def permute(dgm):
    # dgm = randomdgms(10)[0]
    tmp = dgm2diag(dgm)  # tmp is array
    sample_pool = [p[0] for p in tmp] + [p[1] for p in tmp]
    np.random.shuffle(sample_pool)
    assert len(sample_pool) % 2 == 0
    diag = []
    for i in range(0, len(sample_pool), 2):
        b, d = sample_pool[i], sample_pool[i+1]
        if abs(b-d) < 1e-3:
            continue
        tmp = tuple((min(b, d), max(b, d)))
        diag.append(tmp)
    return diag2dgm(diag)

def permute_dgms(dgms):
    res = []
    for dgm in dgms:
        res.append(permute(dgm))
    return res

def fake_diagram(g, cardinality = 2, attribute='deg', seed=42, true_dgm = 'null'):
    random.seed(seed)
    sample_pool = nx.get_node_attributes(g, attribute).values()

    if true_dgm!='null':
        tmp = dgm2diag(true_dgm) # tmp is array
        sample_pool = [p[0] for p in tmp] + [p[1] for p in tmp]

    try:
        sample = random.choice(sample_pool, size=2*cardinality, replace=False)
    except:
        sample = random.choice(sample_pool, size=2 * cardinality, replace=True)
    assert set(sample).issubset(set(sample_pool))

    dgm = []
    for i in range(0, len(sample),2):
        x_ = sample[i]
        y_ = sample[i+1]
        dgm.append((min(x_, y_), max(x_, y_)+1e-3))
    return d.Diagram(dgm)

def fake_diagrams(graphs_, dgms, true_dgms = ['null']*10000, attribute='deg', seed=45):
    fake_dgms = []
    for i in range(len(graphs_)):
        cardinality = len(dgms[i])
        if len(graphs_[i])==0:
             fake_dgms.append(d.Diagram([(0,0)]))
             continue
        tmp_dgm = fake_diagram(graphs_[i][0], cardinality = cardinality, attribute=attribute, seed=seed, true_dgm=true_dgms[i])
        fake_dgms.append(tmp_dgm)
    return fake_dgms
