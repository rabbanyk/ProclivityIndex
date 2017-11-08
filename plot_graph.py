import math
import logging
import numpy as np
import numpy.linalg as LA
import networkx as nx
import networkx.linalg.graphmatrix as gm
import matplotlib 
import matplotlib.pyplot as plt
from pylab import *

FORMAT = "[%(lineno)s:%(funcName)20s()]\n %(message)s"
logging.basicConfig(format=FORMAT, level=logging.ERROR)


def printlatex(A):
    print latex(Matrix(A),mode='inline')

def draw_clustered_graph(G,U, pos, draw_edge_wights = False, draw_graph = True):
    """
    Visualizes the given clustering of a graph.
    
    Parameters
    ----------
    G: A networkx graph 
    U: numpy matrix
        A nxk matrix representing a clustering, where n is the number of data points and k is the number of clusters in U, 
        so that U_{ik} shows if the ith data-point belongs to the kth cluster in U.   
    
    pos: A Python dictionary (optional)
        A dictionary of positions for graph nodes keyed by node

        
    Returns
    -------
    None
    
    Examples
    -------
    >>> import numpy as np
    >>> import networkx as nx
    >>> from pylab import *
    
   >>> np.matrix(  [[0,1,1,0,0,0,0,0,0,0],
                    [1,0,0,1,0,0,0,0,0,0],
                    [1,0,0,1,0,0,0,1,0,0],
                    [0,1,1,0,1,1,0,0,0,0],
                    [0,0,0,1,0,1,1,0,0,0],
                    [0,0,0,1,1,0,1,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0,1,1],
                    [0,0,0,0,0,0,0,1,0,1],
                    [0,0,0,0,0,0,0,1,1,0]])
            
    >>> G = nx.from_numpy_matrix(A)
    >>> U = np.matrix([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0]])
    >>> figure(figsize=(10, 4))
    >>> draw_clustered_graph(G,U, nx.spring_layout(G))
    >>> show()
            
    """
    
    color_list = ['b','g','r','m','y','c']
    color_list =['#9999FF', '#FF9999', '#66FF99','#FF99FF', '#FFFF99', '#E0E0E0','#FF9933','#FF3333','#33FF99','#33FFFF','#3333FF']

    if pos==None:
        pos = nx.spring_layout(G)
        print '----- positions to repeat the layout:: \n ',pos
    
    maxW = 0.0
    for (i,j, w) in G.edges(data=True):
        maxW = max(maxW,w['weight'] )
    if U is not None:
        UG = nx.from_numpy_matrix( U.dot(U.T))
        maxU = U.max()
        # draw clusters
        t =6.2
        for c in range(U.shape[1]):
            nsizes = [ int(t**4 * U[i,c]/maxU) for i in range(U.shape[0])]
            esizes = [ ((U[i,c]*U[j,c])*t**2.0/maxU**2) for (i,j, w) in UG.edges(data=True)]
            
            try:
                nx.draw(UG, pos = pos, node_size=nsizes, style ='solid' ,\
                        #labels ='.',\
                        node_color = color_list[c],edge_color =color_list[c],\
                        linewidths=0, width= esizes, alpha =.5)#, alpha =1
            except:
                pass
    try:
        f= lambda w: np.log(w/maxW +1)*w*3/maxW +1  if draw_graph else 0 
        nx.draw(G, pos = pos, node_color = 'w', alpha =1 ,  linewidths=1, width= [f(w['weight']) for (i,j, w) in G.edges(data=True)])
    except:
        pass
    
    if draw_edge_wights:
        edge_lbls = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos = pos, label_pos=0.5 ,edge_labels =edge_lbls)
   
    return pos

def get_nodes_shaped_based_on_atts(G, attribute_dict):
    colors ={}
    val_to_color = {}
    for att, vals in attribute_dict.items():
        colors[att] = {} 
        dist_colors =  's^>v<odph8s^>v<odph8'
        val_to_color[att] = {vals[i]:dist_colors[i] for i in range(len(vals))}

    for n, atts in G.nodes_iter(data=True):
        for att, val in atts.items():
            if att in colors and val in attribute_dict[att]:
                colors[att][n] = val_to_color[att][val]

    return colors, val_to_color

def get_nodes_colored_based_on_atts(G, attribute_dict):
    colors ={}
    val_to_color = {}
    for att, vals in attribute_dict.items():
        colors[att] = {} 
        dist_colors =  cm.get_cmap('jet', len(vals)) 
        val_to_color[att] = {vals[i]:dist_colors(i)[:3] for i in range(len(vals))}

    for n, atts in G.nodes_iter(data=True):
        for att, val in atts.items():
            if att in colors and val in attribute_dict[att]:
                colors[att][n] = val_to_color[att][val]

    return colors, val_to_color

def draw_attributed_graph(G, attribute_dict , pos= None, print_pos= True, filename = None, block = True):
    figure(figsize=(5, 5))
    if pos==None:
        pos = nx.spring_layout(G)
        if print_pos: 
            print '----- positions to repeat the layout:: \n ',pos
  
    if attribute_dict is not None:
        for att in attribute_dict: 
            if att is not 'shape':
                colors, val_to_color = get_nodes_colored_based_on_atts(G, attribute_dict)
                shapes, val_to_shape = get_nodes_shaped_based_on_atts(G, attribute_dict)
                ncolors = [colors[att][n] if n in colors[att] else (0,0,0) for n in G]
                if not 'shape' in attribute_dict:
                    nx.draw(G , with_labels=True, pos = pos , node_size=1000, edge_color = 'k' , linewidths=2,  width= 4, alpha =.9, node_color = ncolors)
                else :
                    nx.draw_networkx_edges(G,  pos = pos ,  edge_color = 'k' ,  width= 4, alpha =1)
                
                if 'shape' in attribute_dict:
                    att2 = 'shape'
                    nshapes = [shapes[att2][n] if n in shapes[att2] else 'o' for n in G]
                    for sh in attribute_dict[att2]:
                        nbunch = [n for n in G if  nshapes[n]==val_to_shape[att2][sh] ]
                        ncolors = [colors[att][n] if n in colors[att] else (0,0,0) for n in nbunch]

                        nx.draw_networkx_nodes(G, nodelist=nbunch, with_labels=True, pos = pos , node_size=1000, linewidths=2,  alpha =.9, 
                                                node_color = ncolors, node_shape= val_to_shape[att2][sh])
                        plt.axis('off')

                if filename is not None : 
                    savefig('results/' + filename + '.pdf', format='pdf')
                    savefig('results/' + filename + '.png', format='png')

                    show(block = block)
                else:
                    show(block = block)


    return pos


def test_clustered_graph_plot():
    A = np.matrix(  [[0,1,1,0,0,0,0,0,0,0],
                    [1,0,0,1,0,0,0,0,0,0],
                    [1,0,0,1,0,0,0,1,0,0],
                    [0,1,1,0,1,1,0,0,0,0],
                    [0,0,0,1,0,1,1,0,0,0],
                    [0,0,0,1,1,0,1,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,1,0,0,0,0,0,1,1],
                    [0,0,0,0,0,0,0,1,0,1],
                    [0,0,0,0,0,0,0,1,1,0]])
            
    G = nx.from_numpy_matrix(A)
    U = np.matrix([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0]])
    figure(figsize=(10, 4))
    draw_clustered_graph(G,U, nx.spring_layout(G))
    show()



def test_attributed_graph_plot():
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (0,3), (2,0), (2,4), (4,5), (5, 6), (4,6),(6,7),(4,7)])
 
    for i in [0,1,2,3]:
        G.node[i]['gender'] = 1
    for i in [4,5,6,7]:
        G.node[i]['gender'] = 2
    
    attribute_dict = {'gender': [1,2]}  

    print draw_attributed_graph(G, attribute_dict) #, nx.spring_layout(G))

if __name__ == '__main__':

    test_attributed_graph_plot()