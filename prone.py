import sys, os, timeit, time 
import math
import numpy as np
import networkx as nx


def quantify_confusion(N, f=lambda x: x * (x - 1) * 0.5, exact=False):
    if len(N)<=0: return 0
    f = np.vectorize(f, otypes=[np.float])
    b2 = np.ones((N.shape[1], 1), dtype='float')
    b1 = np.ones((1, N.shape[0]), dtype='float')
    m1 = b1.dot(N)
    m2 = N.dot(b2)
    m = b1.dot(N).dot(b2)
    s1 = (b1.dot(f(m2)))
    s2 = (f(m1).dot(b2))
    n = f(m)
    I = (b1.dot(f(N)).dot(b2))
    if exact: E = (b1.dot(f(m2).dot(f(m1)) / f(m))).dot(b2)
    else: E = (b1.dot(f(m2.dot(m1) / m))).dot(b2)
    AG = float((s1 + s2 - 2 * I) / ((s1 + s2) - 2 * E))
    return 1 - AG 

def nmi(N):
    b2 = np.ones((N.shape[1], 1), dtype='float')
    b1 = np.ones((1, N.shape[0]), dtype='float')
    n = b1.dot(N).dot(b2)
    nv = b1.dot(N)
    nu = N.dot(b2)
    f = np.vectorize(lambda x: x * math.log(x) if x > 0 else 0, otypes=[np.float])
    Hu = -(b1.dot(f(nu / n)))
    Hv = -(f(nv / n).dot(b2))
    Huv = -(b1.dot(f(N / n)).dot(b2))
    Iuv = Hu + Hv - Huv

    vi = float((2 * Huv - (Hu + Hv)) / math.log(n))
    nmi_sum = float(2 * Iuv / (Hu + Hv))  # VM in sklearn
    nmi_sqrt = float(Iuv / math.sqrt(Hu * Hv))  # NMI in sklearn

    return np.array([vi, nmi_sum, nmi_sqrt])


def ProNe_l(e): return quantify_confusion(e, f=lambda x: x * math.log(x) if x > 0 else 0)
def ProNe_2(e): return quantify_confusion(e, f=lambda x: x * x )
def ProNe_3(e): return quantify_confusion(e, f=lambda x: x ** 3 ) 

def ass_ind(e, normalize = True):
    if normalize: e = e/np.sum(e)
    s = np.sum(e.dot(e))
    t = np.trace(e)
    return (t - s) / (1 - s)


def ass_ind_2(M, normalize = True):
    if normalize and M.sum() != 1.0: M=M/float(M.sum())
    M = np.asmatrix(M)
    s = (M * M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return float(r)


def node_attribufe_xy(G, attribute1, attribute2, nodes=None):
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    node = G.node
    for u, nbrsdict in G.adjacency_iter():
        if u not in nodes:
            continue
        uattr = node[u].get(attribute1, None)
        if G.is_multigraph():
            for v, keys in nbrsdict.items():
                vattr = node[v].get(attribute2, None)
                for k, d in keys.items():
                    yield (uattr, vattr)
        else:
            for v, eattr in nbrsdict.items():
                vattr = node[v].get(attribute2, None)
                yield (uattr, vattr)


def mixing_dict(xy, normalized=False):
    d = {}
    psum = 0.0
    for x, y in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + 1
        psum += 1

    if normalized:
        for k, jdict in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d

def wrapper(func, *args, **kwargs):
    def wrapped():
        e = attribute_mixing_matrix(*args, **kwargs)
        return func(e)
    return wrapped

def node_attribute_xy(G, attribute1,attribute2, nodes=None, missing_val =0):
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    node = G.node 
    for u,nbrsdict in G.adjacency_iter():
        if u not in nodes:
            continue
        uattr = node[u].get(attribute1,None)
        if G.is_multigraph():
            for v,keys in nbrsdict.items():
                vattr = node[v].get(attribute2,None)                
                for k,d in keys.items():
                    if uattr != missing_val and vattr != missing_val: yield (uattr, vattr)
        else:
            for v,eattr in nbrsdict.items():
                vattr = node[v].get(attribute2,None)
                # print (uattr, vattr)
                if uattr != missing_val and vattr != missing_val: 
                    # print '--- ', (uattr, vattr)
                    yield (uattr, vattr)


def attribute_mixing_dict(G, attribute1, attribute2, nodes=None, normalized=False, missing_val = 0):
    xy_iter = node_attribute_xy(G, attribute1, attribute2, nodes, missing_val)
    return mixing_dict(xy_iter, normalized=normalized)

def dict_to_numpy_array(d,mapping=None, square = True):
    """Convert a dictionary of dictionaries to a 2d numpy array  with optional mapping.
    """
    import numpy
    if mapping is None:
        s=set(d.keys())
        s2 = set()
        for k,v in d.items():
            if square: s.update(v.keys())
            else: s2.update(v.keys())
        mapping=dict(zip(s,range(len(s))))
        mapping2=mapping if square else dict(zip(s2,range(len(s2))))

    a = numpy.zeros((len(mapping), len(mapping2)))
    for k1, i in mapping.items():
        for k2, j in mapping2.items():
            try:
                a[i,j]=d[k1][k2]
            except KeyError:
                pass
    return a


def attribute_mixing_matrix(G, attribute1, attribute2, nodes=None, mapping=None, normalized=True, missing_val=0):
    d = attribute_mixing_dict(G, attribute1, attribute2, nodes, missing_val= missing_val)
    a = dict_to_numpy_array(d, mapping=mapping, square = False)
    if normalized: a = a / a.sum()
    return a

def get_assors(G, att, vals):
    C = [[] for i in range(len(vals))]
    for n,d in G.nodes_iter(data=True): 
        C[vals.index(d[att])].append(n)
    q = Q(G, C)
    return {'Q': q[0], 'r': q[1]}


def get_all(G,attribute_dict, att, att2=None, print_mixing = False, normalized= False, times  = False , missing_val= 0):
    if att2 is None: att2 = att
    e = attribute_mixing_matrix(G, att, att2, normalized=normalized, missing_val= missing_val)   
    if times : 
        wrapped = wrapper(ProNe_2, G, att, att2, normalized=normalized)
        return timeit.timeit(wrapped,number =2)
    res =  { 'ProNe_l': ProNe_l(e), 'ProNe_2': ProNe_2(e), 'ProNe_3': ProNe_3(e)} 
   

    if print_mixing : print e
    if att2==att:
        assors = get_assors(G, att, attribute_dict[att])
        for assor in assors:
            res[assor] = assors[assor]
    return res 



def Q(G, C):
    w = 0.0
    e = 0.0
    m =  nx.number_of_edges(G)
    for c in C:
        for i in c:
            for j in c:
                if i!=j and j in G.neighbors(i): w += 1 
                e += (G.degree(i)*G.degree(j)*1.0/(2*m))

    q = (w-e)/ (2*m)
    n_q = (w-e)/(2*m-e)

    return q, n_q


