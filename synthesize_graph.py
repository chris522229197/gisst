# Generate a synthetic graph for experiments

import os
import sys
import argparse
import numpy as np
import pickle
from networkx.readwrite import gpickle
import torch
from torch_geometric.data import Data
import sig.utils.synthetic_graph as sg
from sig.utils import graph_utils
from sig.utils import pyg_utils


def syn_ba_house(
    ba_size=300,
    num_houses=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a graph with a Barabasi-Albert base with house motifs, with random edges
    added.

    Args:
        ba_size: number of nodes in the Barabasi-Albert base
        num_houses: number of house motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the BA-house graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'ba',
        {'n': ba_size, 'm': 5},
        [['house', {}]] * num_houses,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'ba_{}_house_{}'.format(ba_size, num_houses)
    return graph, roles, name

def syn_ba_comm(
    ba_size=300,
    num_houses=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a union of two BA-house graphs, with random edges added.

    Args:
        ba_size: number of nodes in the Barabasi-Albert base
        num_houses: number of house motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the BA-community graph
        roles: list for the node roles
        name: str for the graph identifier    
    """
    graph_1, roles_1, _ = syn_ba_house(
        rand_edge_p=0,
        start=0,
        role_start=0
    )
    graph_2, roles_2, _ = syn_ba_house(
        rand_edge_p=0, 
        start=graph_1.number_of_nodes(), 
        role_start=max(roles_1) + 1
    )
    graph = sg.rand_join_graphs(
        graph_1, 
        graph_2, 
        ba_size
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    roles = roles_1 + roles_2
    name = 'ba_{}_house_{}_comm'.format(ba_size, num_houses)
    return graph, roles, name

def syn_tree_cycle(
    tree_h=8,
    num_cycles=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a graph with a tree base with cycle motifs, with random edges added.

    Args:
        tree_h: height of the tree base
        num_cycles: number of cycle motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the tree-cycle graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'tree',
        {'h': tree_h, 'r': 2},
        [['cycle', {'num_nodes': 6}]] * num_cycles,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'tree_{}_cycle_{}'.format(tree_h, num_cycles)
    return graph, roles, name

def syn_tree_grid(
    tree_h=8,
    num_grids=80,
    rand_edge_p=0.1,
    **kwargs 
):
    """
    Synthesize a graph with a tree base with grid motifs, with random edges added.

    Args:
        tree_h: height of the tree base
        num_grids: number of grid motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the tree-grid graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'tree',
        {'h': tree_h, 'r': 2},
        [['grid', {'dim': 3}]] * num_grids,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'tree_{}_grid_{}'.format(tree_h, num_grids)
    return graph, roles, name

def syn_tree_house(
    tree_h=8,
    num_houses=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a graph with a tree base with house motifs, with random edges added.

    Args:
        tree_h: height of the tree base
        num_houses: number of house motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the tree-house graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'tree',
        {'h': tree_h, 'r': 2},
        [['house', {}]] * num_houses,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'tree_{}_house_{}'.format(tree_h, num_houses)
    return graph, roles, name

def syn_ba_cycle(
    ba_size=300,
    num_cycles=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a graph with a Barabasi-Albert base with cycle motifs, with random edges
    added.

    Args:
        ba_size: number of nodes in the Barabasi-Albert base
        num_cycles: number of cycle motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the BA-house graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'ba',
        {'n': ba_size, 'm': 5},
        [['cycle', {'num_nodes': 6}]] * num_cycles,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'ba_{}_cycle_{}'.format(ba_size, num_cycles)
    return graph, roles, name

def syn_ba_grid(
    ba_size=300,
    num_grids=80,
    rand_edge_p=0.1,
    **kwargs
):
    """
    Synthesize a graph with a Barabasi-Albert base with grid motifs, with random edges
    added.

    Args:
        ba_size: number of nodes in the Barabasi-Albert base
        num_grids: number of grid motifs
        rand_edge_p: proportion of random edges to add

    Return:
        graph: a networkx Graph for the BA-house graph
        roles: list for the node roles
        name: str for the graph identifier
    """
    graph, roles, _ = sg.build_base_motif(
        'ba',
        {'n': ba_size, 'm': 5},
        [['grid', {'dim': 3}]] * num_grids,
        **kwargs
    )
    graph = sg.add_rand_edges(graph, rand_edge_p)
    name = 'ba_{}_grid_{}'.format(ba_size, num_grids)
    return graph, roles, name
    
def syn_node_feat(
    roles, 
    num_imp_feat=40,
    p_unimp_feat=0.25,
    sigma_scale=1
):
    """
    Synthesize node features for a graph.

    Args:
        roles: list of node roles corresponding to the nodes in graph
        num_imp_feat: number of important node features based on the node roles
        p_unimp_feat: proportion of noisy unimportant node features with respect to 
                      num_imp_feat

    Return:
        node_feat: numpy array of the node features
        node_feat_imp: numpy array of the important node feature indices
    """
    node_feat = sg.gen_node_feat(
        roles, 
        num_imp_feat, 
        mu_scale=1,
        sigma_scale=sigma_scale
    )
    node_feat = sg.add_rand_node_feat(
        node_feat, 
        p_unimp_feat,
        mu=0,
        sigma=sigma_scale
    )
    node_feat_imp = np.arange(num_imp_feat)
    return node_feat, node_feat_imp

def write_graph_dist(
    outfile,
    nodes, 
    edges,
    y
):
    """
    Write graph data distribution to a text tile.

    Args:
        outfile: file object for writing
        nodes: list of nodes in the graph
        edges: list of edges in the graph
        y: list of node labels
    """
    total = len(y)
    count = {}
    for label in y:
        if label not in count.keys():
            count[label] = 0
        count[label] += 1
    labels= list(count.keys())
    labels.sort()

    outfile.write('Num nodes: {}\n'.format(len(nodes)))
    outfile.write('Num edges: {}\n'.format(len(edges)))
    for label in labels:
        label_count = count[label]
        outfile.write('%d: %d (%0.3f)\n' % (label, label_count, label_count / total))

def main():
    parser = argparse.ArgumentParser(
        description="generate a synthetic graph with a base and attached motifs"
    )

    parser.add_argument(
        'outdir',
        type=str,
        help='output directory with trailing /'
    )
    parser.add_argument(
        '--topology', 
        type=str, 
        choices=[
            'ba_house', 
            'ba_comm', 
            'tree_cycle', 
            'tree_grid',
            'tree_house',
            'ba_cycle',
            'ba_grid'
        ],
        default='ba_house',
        help='topology for the synthetic graph',
        dest='topology'
    )
    parser.add_argument(
        '--np-seed',
        type=int,
        default=123,
        help='random seed for numpy',
        dest='np_seed'
    )
    parser.add_argument(
        '--fold-seed',
        type=int,
        default=456,
        help='random seed for splitting the graph into train, val, and test folds',
        dest='fold_seed'
    )

    args = parser.parse_args()
    print('Running {} with arguments'.format(sys.argv[0]))
    for arg in vars(args):
        print('\t{}={}'.format(arg, getattr(args, arg)))

    print('Synthesizing the {} graph...'.format(args.topology))
    np.random.seed(args.np_seed)
    graph, roles, name = eval('syn_' + args.topology)()
    if len(set(roles)) == 2:
        sigma_scale = 0.5
    else:
        sigma_scale = 0.15
    node_feat, node_feat_imp = syn_node_feat(roles, sigma_scale=sigma_scale)
    node_train, node_val, node_test, \
        edge_train, edge_val, edge_test, \
        y_train, y_val, y_test = graph_utils.split_graph_train_val_test(
            graph, 
            roles, 
            seed=args.fold_seed, 
            both_dir=True
        )
    
    print('Formatting the data for PyTorch Geometric...')
    data = Data()
    data.x = torch.Tensor(node_feat)
    data.y = torch.LongTensor(roles)
    edge_index = pyg_utils.get_pyg_edge_index(graph)
    data.edge_index = edge_index
    edge_tuples = [
        (edge_index[0, i].item(), edge_index[1, i].item()) \
            for i in range(edge_index.shape[1])
    ]
    graph_nodes = list(graph.nodes)
    data = pyg_utils.process_pyg_data(
        data, 
        graph_nodes, 
        edge_tuples, 
        node_train, 
        edge_train, 
        'train'
    )
    data.y_train = data.y[data.node_mask_train]

    data = pyg_utils.process_pyg_data(
        data, 
        graph_nodes, 
        edge_tuples, 
        node_val, 
        edge_val, 
        'val'
    )
    data.y_val = data.y[data.node_mask_val]

    data = pyg_utils.process_pyg_data(
        data, 
        graph_nodes, 
        edge_tuples, 
        node_test, 
        edge_test, 
        'test'
    )
    data.y_test = data.y[data.node_mask_test]

    imp_node_feat_mask = pyg_utils.get_mask(
        np.arange(node_feat.shape[1]), 
        node_feat_imp, 
        pyg=True
    )
    imp_node_feat_mask = imp_node_feat_mask.expand(node_feat.shape[0], -1)
    data.imp_node_feat_mask = imp_node_feat_mask
    data.imp_edge_mask = pyg_utils.get_pyg_imp_edge_mask(graph, data.edge_index)

    sub_outdir = args.outdir + name + '/'
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)

    print('Saving the output to {}...'.format(sub_outdir))
    gpickle.write_gpickle(graph, sub_outdir + 'graph.gpkl')
    torch.save(data, sub_outdir + 'data.pyg.pkl')

    print('Saving the log file to {}...'.format(sub_outdir))
    with open(sub_outdir + 'log.txt', 'w') as f:
        f.write('{}\n'.format(sys.argv[0]))
        for arg in vars(args):
            f.write('{}={}\n'.format(arg, getattr(args, arg)))
        f.write('\n')
        for split in ['train', 'val', 'test']:
            f.write('{}\n'.format(split))
            write_graph_dist(
                f,
                eval('node_' + split),
                eval('edge_' + split),
                eval('y_' + split)
            )
            f.write('\n')
    print('Finished running {}'.format(sys.argv[0]))


if __name__ == '__main__':
    main()