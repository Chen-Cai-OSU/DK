""" Implement different filtration stradegies for graphs """
import networkx as nx
import graphonlib
import numpy as np
import dionysus as d
from dgms.format import dgms2diags, dgm2diag, diag2dgm, res2dgms, diags2dgms
from dgms.format import diag_check, flip_dgm
from dgms.stat import print_dgm
from dgms.compute import print_f
from dgms.format import export


class fil_stradegy():
    def __init__(self, g, fil = 'combined', node_fil = 'sub', edge_fil = 'sup', **kwargs):
        """

        :param g: networkx graph
        :param fil: node, edge, or combined
        :param node_fil: used only when fil is node or combined.
        :param edge_fil: used only when fil is edge or combined
        :param kwargs:  kwargs: nodefeat is A np.array of shape (n,1)
                        kwargs: edgefeat is a n*n matrix or other data structure that supports x[u][v]

        """
        self.g = g
        self.n = len(g)
        self.fil = fil
        self.node_fil = node_fil
        self.edge_fil = edge_fil
        self.nodefeat_ = kwargs.get('nodefeat')
        self.edgefeat_ = kwargs.get('edgefeat')

    def nodefeat(self, func = None, nodefeat= None):
        # set node feat for graph

        if func == 'random':
            self.nodefeat_ = np.random.random((self.g, 1))
        else:
            self.nodefeat_ = nodefeat

    def edgefeat(self, func = None, **kwargs):
        # implement a few common edge vals

        edgefeat = np.zeros((self.n, self.n))
        if func == 'jaccard':
            edgefeat= nx.jaccard_coefficient(self.g, list(self.g.edges))
        elif func == 'edge_prob':
            adj = nx.adjacency_matrix(self.g).todense()
            edgefeat = graphonlib.smoothing.zhang.smoother(adj, h=kwargs.get('h', 0.3))  # h : neighborhood size parameter. Example: 0.3 means to include
        else:
            edgefeat = kwargs['edgefeat']

        self.edgefeat_ = edgefeat
        self.edge_fil = kwargs['edge_fil']

    def build_fv(self, **kwargs):
        if self.fil == 'combined':
            assert self.nodefeat_ is not None
            assert self.edgefeat_ is not None

            for u, v in self.g.edges():
                self.g[u][v]['fv'] = self.edgefeat_[u][v]
            for n in self.g.nodes():
                self.g.node[n]['fv'] = self.nodefeat_[n, 0].astype(float)

        if self.fil == 'node':
            # sublevel fil
            assert self.nodefeat_ is not None
            for n in self.g.nodes():
                self.g.node[n]['fv'] = self.nodefeat_[n, 0].astype(float)

            if self.node_fil == 'sub':
                op = lambda x: max(x)
            elif self.node_fil == 'sup':
                op = lambda x: min(x)
            else:
                raise Exception('Error in node fil')

            for u, v in self.g.edges():
                self.g[u][v]['fv'] = op([self.g.node[u]['fv'], self.g.node[v]['fv']])

        if self.fil == 'edge':
            assert self.edgefeat_ is not None
            for u, v in self.g.edges():
                self.g[u][v]['fv'] = self.edgefeat_[u][v]

            for n in self.g.nodes():
                nbrs = list(nx.neighbors(self.g, n))
                vals = [self.edgefeat_[n][nbr] for nbr in nbrs]
                if len(vals)<1: vals = [0]
                if self.edge_fil == 'sup':
                    self.g.node[n]['fv'] = max(vals)  # min or max? should be max for superlevel and min for sublevel.
                elif self.edge_fil == 'sub':
                    self.g.node[n]['fv'] = min(vals)
                else:
                    raise Exception('Error in edge fil')

        return self.g

class graph2dgm():
    def __init__(self, g, **kwargs):
        self.graph = nx.convert_node_labels_to_integers(g)
        self.nodefil = kwargs.get('nodefil', 'sub')
        self.edgefil = kwargs.get('edgefil', 'sup')

    def check(self):
        for n in self.graph.nodes():
            assert 'fv' in self.graph.node[n]

    def get_simplices(self, gi, key='fv'):
        """Used by get_diagram function"""
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert key in gi.node[list(gi.nodes())[0]].keys()
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = list()
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            tup = ([u, v], data[key])
            simplices.append(tup)
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            tup = ([v], data[key])
            simplices.append(tup)
        return simplices

    def del_inf(self, dgms):
        # remove inf
        dgms_list = [[], []]
        for i in range(2):
            pt_list = list()
            for pt in dgms[i]:
                if (pt.birth == float('inf')) or (pt.death == float('inf')):
                    pass
                else:
                    pt_list.append(tuple([pt.birth, pt.death]))
            diagram = d.Diagram(pt_list)
            dgms_list[i] = diagram
        return dgms_list

    def compute_PD(self, simplices, sub=True, inf_flag='False', zigzag = False):
        def cmp(a, b):
            return (a > b) - (a < b)
        def compare(s1, s2, sub_flag=True):
            if sub_flag == True:
                if s1.dimension() > s2.dimension():
                    return 1
                elif s1.dimension() < s2.dimension():
                    return -1
                else:
                    return cmp(s1.data, s2.data)
            elif sub_flag == False:
                return -compare(s1, s2, sub_flag=True)
        def zigzag_less(x, y):
            # x, y are simplex
            dimx, datax = x.dimension(), x.data
            dimy, datay = y.dimension(), y.data
            if dimx == dimy == 0:
                return datax <= datay
            elif dimx == dimy == 1:
                return datax >= datay
            else:
                return dimx < dimy

        def zigzag_op(nodefil = 'sub', edgefil = 'sup'):

            if nodefil == 'sub':
                node_op = lambda x, y: x < y
            elif nodefil == 'sup':
                node_op = lambda x, y: x > y
            else:
                raise Exception('node fil Error')

            if edgefil == 'sub':
                edge_op = lambda x, y: x <= y
            elif edgefil == 'sup':
                edge_op = lambda x, y: x > y
            else:
                raise Exception('edge fil Error')

            # x, y are simplex
            def op(x, y):
                dimx, datax = x.dimension(), x.data
                dimy, datay = y.dimension(), y.data
                if dimx == dimy == 0:
                    return node_op(datax, datay)
                elif dimx == dimy == 1:
                    return edge_op(datax, datay)
                else:
                    return dimx < dimy
            return op

        f = d.Filtration()
        for simplex, time in simplices:
            f.append(d.Simplex(simplex, time))

        if not zigzag:
            f.sort() if sub else f.sort(reverse=True)
        else:
            f.sort(zigzag_op(self.nodefil, self.edgefil), reverse=True)
            # print('After zigzag\n')
            # print_f(f)

            # test case
            # simplices = [([2], 4), ([1, 2], 5), ([0, 2], 6),([0], 1), ([1], 2), ([0, 1], 3)]
            # f = d.Filtration()
            # for vertices, time in simplices:
            #     f.append(d.Simplex(vertices, time))
            # f.append(d.Simplex(vertices, time))
            # f.sort(cmp=zigzag_operator(nodefil = 'sub', edgefil = 'sup'),reverse=True)
            # print_f(f)

        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)

        if inf_flag == 'False':
            dgms = self.del_inf(dgms)
        # for some degenerate case, return dgm(0,0)
        if (dgms == []) or (dgms == None):
            return d.Diagram([[0,0]])
        return dgms

    def get_diagram(self, g, key='fv', subflag = 'True', one_homology_flag=False, parallel_flag = False, zigzag = False):
        """

        :param g: networkx graph with fv computed on each node and edge
        :param key: fv. This is the key to access filtration function value
        :param subflag: 'True' if sub level filtration used. 'False' if superlevel filtration used.
        :param one_homology_flag: ignore for now.
        :param parallel_flag: ignore for now.
        :param zigzag: Set to be true if you want to use combined filtration. (set filtration for nodes and edges seprately,
                instead of using node filtration or edge filtration.)
        :return: Persistence diagram
        """

        # only return 0-homology of sublevel filtration TODO: include one homology
        # type can be tuple or pd. tuple can be parallized, pd cannot.
        g = nx.convert_node_labels_to_integers(g)
        simplices = self.get_simplices(g, key = key)
        if one_homology_flag:
            epd_dgm = self.epd(self, g, pd_flag=False)[1]
            epd_dgm = self.post_process(epd_dgm)
            return epd_dgm

        super_dgms = self.compute_PD(simplices, sub=False)
        sub_dgms = self.compute_PD(simplices, sub=True) if not zigzag else self.compute_PD(simplices, zigzag=True)

        _min = min([g.node[n][key] for n in g.nodes()])
        _max = max([g.node[n][key] for n in g.nodes()])+ 1e-5 # avoid the extra node lies on diagonal
        p_min = d.Diagram([(_min, _max)])
        p_max = d.Diagram([(_max, _min)])

        sub_dgms[0].append(p_min[0])
        super_dgms[0].append(p_max[0])

        if subflag=='True':
            return sub_dgms[0] if not parallel_flag else dgm2diag(sub_dgms[0])
        elif subflag=='False':
            return super_dgms[0] if not parallel_flag else dgm2diag(super_dgms[0])
        else:
            raise Exception('subflag can be either True or False')

    def epd(self, g__, pd_flag=False, debug_flag=False):
        w = -1
        values = nx.get_node_attributes(g__, 'fv')
        simplices = [[x[0], x[1]] for x in list(g__.edges)] + [[x] for x in g__.nodes()]
        up_simplices = [d.Simplex(s, max(values[v] for v in s)) for s in simplices]
        down_simplices = [d.Simplex(s + [w], min(values[v] for v in s)) for s in simplices]
        if pd_flag == True:
            down_simplices = []  # mask the extended persistence here

        up_simplices.sort(key=lambda s1: (s1.dimension(), s1.data))
        down_simplices.sort(reverse=True, key=lambda s: (s.dimension(), s.data))
        f = d.Filtration([d.Simplex([w], -float('inf'))] + up_simplices + down_simplices)
        m = d.homology_persistence(f)
        dgms = d.init_diagrams(m, f)
        if debug_flag == True:
            print('Calling compute_EPD here with success. Print first dgm in dgms')
            print_dgm(dgms[0])
        return dgms

    def post_process(self, dgm, debug_flag=False):
        if len(dgm) == 0:
            return d.Diagram([(0, 0)])
        for p in dgm:
            if p.birth == np.float('-inf'):
                p.birth = 0
            if p.death == np.float('inf'):
                p.death = 0
        if debug_flag == True:
            print('Before flip:'),
            print_dgm(dgm)
        dgm = flip_dgm(dgm)
        if debug_flag == True:
            print('After:'),
            print_dgm(dgm)
        return dgm


if __name__ == '__main__':
    np.random.seed(42)
    n_node = 20
    g = nx.random_geometric_graph(n_node, 0.5, seed=42)
    assert nx.is_connected(g)

    # node feat example
    nodefeat = np.random.random((n_node, 1))
    fil = fil_stradegy(g, fil='node', node_fil='sub', nodefeat = nodefeat)
    g = fil.build_fv()
    for u, v in g.edges():
        assert g[u][v]['fv'] == max(g.node[u]['fv'], g.node[v]['fv'])
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)

    # edge feat example
    fil = fil_stradegy(g, fil='edge')
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    g = fil.build_fv()
    for u in g.nodes():
        nbrs = nx.neighbors(g, u)
        nbredgevals = []
        for v in nbrs:
            nbredgevals.append(g[u][v]['fv'])
        # print(g.node[u]['fv'], min(nbredgevals),max(nbredgevals))
        assert g.node[u]['fv'] == max(nbredgevals)
    x = graph2dgm(g)
    diagram = x.get_diagram(g, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=False)

    # combined fil -- one way
    fil = fil_stradegy(g, fil='combined', nodefeat=nodefeat)
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    g1 = fil.build_fv()
    x = graph2dgm(g1, nodefil='sub', edgefil='sup')
    diagram = x.get_diagram(g1, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)

    # combined fil -- anathor way. Almost the same with the previous example.
    fil = fil_stradegy(g, fil='combined')
    fil.edgefeat(func='edge_prob', edge_fil='sup')
    fil.nodefeat(nodefeat)
    g2 = fil.build_fv()
    x = graph2dgm(g2, nodefil='sub', edgefil='sup')
    dgm = x.get_diagram(g2, key='fv', subflag='True', one_homology_flag=False, parallel_flag=False, zigzag=True)
    print_dgm(dgm)
    export(dgm, dir='./RNA/', filename='dgm.csv')

