import dionysus as d
import sys
import os
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
import time
import csv
from sklearn.preprocessing import normalize
def diag2array(diag):
    return np.array(diag)

def array2diag(array):
    res = []
    n = len(array)
    for i in range(n):
        p = [array[i,0], array[i,1]]
        res.append(p)
    return res

def dgm2diag(dgm):
    assert str(type(dgm)) == "<class 'dionysus._dionysus.Diagram'>"
    diag = list()
    for pt in dgm:
        if str(pt.death) == 'inf':
            diag.append([pt.birth, float('Inf')])
        else:
            diag.append([pt.birth, pt.death])
    return diag
def dgms2diags(dgms):
    t0 = time.time()
    diags = []
    for i in range(len(dgms)):
        diags.append(dgm2diag(dgms[i]))
    print ('Finish converting dgms to diags in %s'%(time.time()-t0))
    return diags

def diag2dgm(diag):
    import dionysus as d
    if type(diag) == list:
      diag = [tuple(i) for i in diag]
    elif type(diag) == np.ndarray:
      diag = [tuple(i) for i in diag] # just help to tell diag might be an array
    dgm = d.Diagram(diag)
    return dgm

def diags2dgms(diags):
    t0 = time.time()
    dgms = []
    for diag in diags:
      dgms.append(diag2dgm(diag))
    print ('Finish converting diags to dgms in %s'%(time.time()-t0))
    return dgms

def diag_check(diag):
    if type(diag)==list and set(map(len, diag)) == {2}:
        return True
    else:
        return False

def res2dgms(res):
    dgms = []
    for diag in res:
        assert diag_check(diag)
        dgms.append(diag2dgm(diag))
    return dgms

def dgms2swdgms(dgms):
    swdgms = []
    for dgm in dgms:
        diag = dgm2diag(dgm)
        swdgms += [np.array(diag)]
    return swdgms

def flip_dgm(dgm):
    # flip dgm from below to above, not vise versa
    for p in dgm:
        if np.float(p.birth) < np.float(p.death):
            assert_dgm_above(dgm)
            return dgm
        assert np.float(p.birth) >= np.float(p.death)
    data = [(np.float(p.death), np.float(p.birth)) for p in dgm]
    return d.Diagram(data)

def normalize_dgm(dgm):
    # dgm = dgms[1]
    diag = np.array(dgm2diag(dgm))
    diag = normalize(diag, axis=0)
    n = len(diag)
    res = []
    for i in range(n):
        res.append(list(diag[i,:]))
    return diag2dgm(res)

def normalize_dgms(dgms):
    res = []
    for dgm in dgms:
        res.append(normalize_dgm(dgm))
    return res

def export(dgm, dir='./', filename='dgm.csv'):
    # dgm = diagram
    diag = dgm2diag(dgm)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(diag)