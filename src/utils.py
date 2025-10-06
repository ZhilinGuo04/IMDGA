import os
import torch
import random
import numpy as np
import copy
import networkx as nx



def set_seed(rand_seed):
    rand_seed = rand_seed if rand_seed >= 0 else torch.initial_seed() % 4294967295  # 2^32-1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
    

def compute_assortativity(graph, attack_nodes, group_nodes, data_y):
    labels = data_y.reshape(-1).cpu().numpy() if data_y.is_cuda else data_y.numpy()
    nx.set_node_attributes(graph, {i: {'label': int(labels[i])} for i in range(len(labels))})

    #  attack_node 
    assortativity_dict = {}

    #  attack_node 
    for attack_node, neighbor_nodes in zip(attack_nodes, group_nodes):
        attack_node = int(attack_node)
        if not neighbor_nodes:
            assortativity_dict[attack_node] = None
            continue


        subgraph = graph.subgraph(neighbor_nodes)
        node_labels = [graph.nodes[n]['label'] for n in subgraph.nodes()]
        if subgraph.number_of_nodes() == 0:
            assortativity_dict[attack_node] = None
            continue

        # （ 'label' ）
        try:
            if len(set(node_labels)) == 1:
                assortativity_dict[attack_node] = 1
            else:
                assortativity = nx.attribute_assortativity_coefficient(subgraph, 'label')
                assortativity_dict[attack_node] = assortativity
        except nx.NetworkXError as e:
            assortativity_dict[attack_node] = None

    return assortativity_dict