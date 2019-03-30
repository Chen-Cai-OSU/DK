import dionysus as d
from dgms.format import dgm2diag, diag2dgm, array2diag
import numpy as np

def upperdiagonal(dgm):
    for p in dgm:
        assert p.birth <= p.death


class dgm_operator(d._dionysus.Diagram):
    def __init__(self, dgms):
        self.dgms = dgms

    def overlay(self):
        diags = []
        for dgm in self.dgms:
            diag = dgm2diag(dgm)
            diags.append(np.array(diag))
        # diags = [np.random.random((3, 2)), np.random.random((2,2))]
        res = np.concatenate(tuple(diags), axis=0) # array
        res = array2diag(res)
        res = diag2dgm(res)
        return res
