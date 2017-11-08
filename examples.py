import sys
import os
import matplotlib  
import matplotlib.pyplot as plt
import networkx as nx
import plot_graph as pltg
from numpy import array
from prone import *

def mesure_and_plot(G, attribute_dict, att = 'color', att2=None, name = 'e', pos = None, print_pos=False, normalize = False):
    if att2 is None: att2= att
    
    print  '---------',name,'----------'
    res = get_all(G,attribute_dict, att, att2, print_mixing = False, normalized= False ) 
    if 'Q' in res:
        print 'Q', '\t\t', '{:.2f}'.format(res['Q'])
        print 'r' , '\t\t', '{:.2f}'.format(res['r'])
    else :
        print 'Q:\t\t x '
        print 'r:\t\t x' 

    for sca in ['ProNe_l','ProNe_2','ProNe_3']: 
        print sca, '\t', '{:.2f}'.format(res[sca]) 
    

    pltg.draw_attributed_graph(G,attribute_dict, pos, print_pos=print_pos, filename= name)



def example_self_proc_1():
    print "example i.1"
    G = nx.MultiGraph()
    G.add_edges_from([(0,1), (0,2), (1,2), (3,4),(4,5), (5,3),(6,7),(6,8),(7,8),(9,10),(9,11),(11,10)   ])
    # print G.number_of_selfloops()
    attribute_dict = {'color': [1,2,3,4]}

    for i in [0,1,2]:        
        G.node[i]['color'] = 1
    for i in [3,4,5]:       
        G.node[i]['color'] = 2
    for i in [6,7,8]:       
        G.node[i]['color'] = 3
    for i in [9,10,11]:     
        G.node[i]['color'] = 4
  
    pos =   {0: array([ 0.8162686,  0.54364317]), 1: array([ 0.80064918,  0.21996616]), 2: array([ 0.96658515,  0.36753988]), 
            3: array([ 0.52827516,  0.12900959]), 4: array([ 0.20051781,  0.17879567]), 5: array([ 0.35928663,  0.        ]), 
            6: array([ 0.2        ,  0.43033347]), 7: array([ 0.00180532,  0.60660009]), 8: array([ 0.20933775,  0.74104649]), 
            9: array([ 0.53169441,  1.2        ]), 10: array([ 0.36126299,  0.95467806]), 11: array([ 0.6986756 ,  0.92379552])}
    
    mesure_and_plot(G, attribute_dict, name = 'e1', pos = pos)

def example_self_proc_2():
    print "example i.2"
    G = nx.Graph()
    G.add_edges_from([(0,3), (0,4), (0,5), (1,3),(1,4), (1,5), (2,3),(2,4), (2,5), (6,9), (7,9), (8,9), (6,10),(7,10), (8,10), (6,11),(7,11), (8,11)])
    attribute_dict = {'color': [1,2,3,4]}

    for i in [0,1,2]:       
        G.node[i]['color'] = 1
    for i in [3,4,5]:       
        G.node[i]['color'] = 2
    for i in [6,7,8]:       
        G.node[i]['color'] = 3
    for i in [9,10,11]:     
        G.node[i]['color'] = 4
    
    pos =   {0: array([ 0, 0]), 1: array([ 0,  0.25]), 2: array([ 0,  0.5]), 
        3: array([ 0.35,  0]), 4: array([ 0.35,  0.25]), 5: array([ 0.35        ,  0.5]), 
        6: array([ 0.75 , 0.]), 7: array([ 0.75 , 0.25]), 8: array([ 0.75 , 0.5]), 
        9: array([ 1,  0.]), 10: array([ 1,  0.25]), 11: array([ 1,  0.5])}
    mesure_and_plot(G, attribute_dict, name = 'e2', pos = pos)


def example_self_proc_3():
    print "example i.3"
    G = nx.Graph()
    G.add_edges_from([(0,9), (0,3), (0,6), (9,3), (6,3),(9,6),
                  (1,4),(1,7), (1,10), (7,10),(7,4), (4,10),
                       (2,5),(2,8), (2,11),(8,5),(8,11),(5,11)   ])
    
    attribute_dict = {'color': [1,2,3,4]}

    for i in [0,1,2]:        
        G.node[i]['color'] = 1
    for i in [3,4,5]:       
        G.node[i]['color'] = 2
    for i in [6,7,8]:       
        G.node[i]['color'] = 3
    for i in [9,10,11]:     
        G.node[i]['color'] = 4

    pos=   {0: array([0, 0.1]), 1: array([0,  0.5]), 2: array([0,  0.9]), 
             3: array([ 0.1,  0]), 4:array([ 0.1,  0.4]), 5: array([ 0.1 ,  0.8]), 
             6: array([ 0.1 , 0.2]), 7: array([ 0.1 , 0.6]), 8: array([ 0.1 , 1]), 
             9: array([ 0.2,  0.1]), 10: array([ 0.2,  0.5]), 11: array([ 0.2,  0.9]),}

    mesure_and_plot(G, attribute_dict, name = 'e3', pos = pos)



def example_cross_proc_1():
    print "example ii.1"
    G = nx.Graph()
    G.add_edges_from([(0,1), (0,2), (0,3), (1,2), (2,3),(1,3), (4,5),(4,6), (4,7), (7,6),(7,5), (5,6),])
    attribute_dict = {'color': [1,2,3,4], 'shape':[1,2]}
    for i in [0,1]:        
        G.node[i]['color'] = 1
    for i in [2,3]:        
        G.node[i]['color'] = 2
    for i in [4,5]:        
        G.node[i]['color'] = 3
    for i in [6,7]:        
        G.node[i]['color'] = 4

    for i in [0,1,2,3]:        
        G.node[i]['shape'] = 1
    for i in [4,5,6,7]:        
        G.node[i]['shape'] = 2
    

    pos=   { 0: array([0, 0.6]), 1: array([ 0.1,  0.6]), 2: array([0,  0.4]), 3: array([ 0.1,  .4]), 
            4:array([ 0,  0.2]), 5: array([ 0.1 ,  0.2]),  6: array([ 0. , 0.]), 7: array([ 0.1 , 0]),}

    mesure_and_plot(G, attribute_dict,att2='shape', name = 'e4', pos = pos)

def example_cross_proc_2():
    print "example ii.2"
    G = nx.Graph()
    G.add_edges_from([(0,1), (0,2), (0,3), (1,2), (2,3),(1,3), (4,5),(4,6), (4,7), (7,6),(7,5), (5,6),])
    attribute_dict = {'color': [1,2,3,4], 'shape':[1,2]}

    for i in [0,1]:        
        G.node[i]['color'] = 1
    for i in [2,3]:        
        G.node[i]['color'] = 2
    for i in [4,5]:        
        G.node[i]['color'] = 3
    for i in [6,7]:        
        G.node[i]['color'] = 4

    for i in [0,2,4,6]:        
        G.node[i]['shape'] = 1
    for i in [1,3,5,7]:        
        G.node[i]['shape'] = 2
    

    pos=   { 0: array([0, 0.6]), 1: array([ 0.1,  0.6]), 2: array([0,  0.4]), 3: array([ 0.1,  .4]), 
            4:array([ 0,  0.2]), 5: array([ 0.1 ,  0.2]),  6: array([ 0. , 0.]), 7: array([ 0.1 , 0]),}

    mesure_and_plot(G, attribute_dict,att2='shape', name = 'e5', pos = pos)



def examples_self_cross():
    print '_'*30
    example_self_proc_1()
    print '_'*30
    example_self_proc_2()
    print '_'*30
    example_self_proc_3()
    print '_'*30
    example_cross_proc_1()
    print '_'*30
    example_cross_proc_2()


font = {'family' : 'serif', 'weight' : 'bold',  'size'   : 32}
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 22})



if __name__ == '__main__':
    examples_self_cross()
 