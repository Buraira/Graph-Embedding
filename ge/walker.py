import itertools
import math
import networkx as nx
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange

from .alias import alias_sample, create_alias_table
from .utils import partition_num
from random import choice


class RandomWalker:
    def __init__(self, G, p=1, q=1, clust={},clust_dic={},metric1={},same_metric1_dic={},metric2={}):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q
        
        #my code
        self.metric2 = metric2
        self.clust = clust
        self.clust_dic = clust_dic
        self.metric1 = metric1
        self.same_metric1_dic=same_metric1_dic
   
    def clust_coeff(self):
        for node,coeff in self.clust.items():
            if coeff in self.clust_dic:
                continue
            else:
                same_clust = {k for k,v in self.clust.items() if v == coeff}
                self.clust_dic[coeff]=same_clust       
        #...
    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
    
    #shortcut_walk
        
    def shortcut_walk(self, walk_length, start_node):
        
        walk = [start_node]
        nodes = list(self.G.nodes)
        
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            rand = random.random()
            
            if (rand <= self.p):
                if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
                else:
                    break
            else:
                walk.append(choice(nodes))
         
        return walk
    
    def similarity_walk(self, walk_length, start_node):
          
        walk = [start_node]
        nodes = list(self.G.nodes)
        listsamecoeff =[]
        #print(self.clust)
        while len(walk) < walk_length:
            
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            rand = random.random()
            
            if (rand <= self.p):
                if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
                else:
                    break
            else:
                #coefficient for cur
                c = [v for k,v in self.clust.items() if k == cur]
                #nodes of same coefficient
                listsamecoeff = [v for k,v in self.clust_dic.items() if k == c[0]]
                n = random.choice(listsamecoeff)
                ran =[x for x in n if x != str(cur)]
                loc = sorted([(k,v) for k,v in self.clust.items()],key=lambda tup: tup[1])

                if not ran:
                    for x in range(len(loc)-1):
                        if (loc[x][0] == cur):
                            ran2 =random.choice([loc[x-1][0],loc[x+1][0]])
                            break
                        elif (x == 0):
                            ran2 = loc[x+1][0]
                            break
                        elif(x == len(loc)-1):
                            ran2 = loc[x-1][0]
                            break
                        else:
                            break
                else:
                    ran2 = random.choice(ran)

             
                walk.append(str(ran2))
        
        return walk


   

    def similarity_walk_m(self, walk_length, start_node):
                 
        walk = [start_node]
        nodes = list(self.G.nodes)
        listsamemet =[]
        
        while len(walk) < walk_length:
            
            cur = walk[-1]
           
            cur_nbrs = list(self.G.neighbors(cur))
            rand = random.random()
            
            if (rand <= self.p):
                if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
                else:
                    break
            else:
                #metric1 for current node 
                d = [v for k,v in self.metric1.items() if k == cur]
             
                #nodes of same metric1
                listsamemet = [v for k,v in self.same_metric1_dic.items() if k == d[0]]
                n = random.choice(listsamemet)
               
                #metric2 for current node         
                met2 = [y for x,y in self.metric2.items() if x == cur ]
                
                r =[x for x in n if x != str(cur)]
                
                #metric2 for the nodes with same metric1
                metric2_d={}
                for x in r:
                    for y,z in self.metric2.items():
                        if (y == x):
                            dr = z    
                    metric2_d.update({x:dr})

                #metric2_d.sort()
                metric2_d_sor = sorted([(k,v) for k,v in metric2_d.items()],key=lambda tup: tup[1])
                
                loc = sorted([(k,v) for k,v in self.metric1.items()],key=lambda tup: tup[1])
           
                if not r:
                    
                    for x in range(len(loc)-1):
                        if (loc[x][0] == cur and x!=0):
                            rand =random.choice([loc[x-1][0],loc[x+1][0]])
                            break
                        elif (x == 0):
                            rand = loc[x+1][0]
                            break
                        elif(x == len(loc)-1):
                            rand = loc[x-1][0]
                            break
                        else:
                            break
                    
                else:
                    
                    flag = [n for n,x in metric2_d.items() if x==met2[0]]
                    flag_metric2 = [x for n,x in metric2_d.items() if x==met2[0]]
                   
                    if len(flag)==1:
                        rand = flag[0]
                    elif len(flag)>1:
                        rand = random.choice(flag)
                    elif (len(flag)==0):
                        n = len(metric2_d_sor)-1
                       
                        for x in range(len(metric2_d_sor)):
                            
                            if (met2[0] < metric2_d_sor[x][1] and x!=n):
                                rand = metric2_d_sor[x][0]
                                break
                            elif(met2[0] < metric2_d_sor[x][1] and x==n):
                                rand = metric2_d_sor[x][0]
                                break
                            elif(met2[0] > metric2_d_sor[x][1] and x==n):
                                rand = metric2_d_sor[x][0]
                                break
                
                walk.append(str(rand))
        
        return walk

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                else:
                    #walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
                    #walks.append(self.similarity_walk(walk_length=walk_length, start_node=v,))
                    #walks.append(self.shortcut_walk(walk_length=walk_length, start_node=v))
                    walks.append(self.similarity_walk_m(walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):

        layers_adj = pd.read_pickle(self.temp_path+'layers_adj.pkl')
        layers_alias = pd.read_pickle(self.temp_path+'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path+'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path+'gamma.pkl')
        walks = []
        initialLayer = 0

        nodes = self.idx  # list(self.g.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma)
            for num in partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()
            if(r < stay_prob):  # same layer
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if(r > p_moveup):
                    if(layer > initialLayer):
                        layer = layer - 1
                else:
                    if((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):

    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return v
