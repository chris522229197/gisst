# Utilies for handling graphs

import os.path as osp
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_networkx
from torch_geometric.utils import subgraph as pyg_subgraph
import torch_geometric.transforms as T
from sig.utils.pyg_utils import pyg_edge_index_to_tuples


def get_subgraph_edges(
    graph, 
    subgraph_nodes, 
    both_dir=True
):
    """
    Get the edges from a subgraph with given nodes.

    Args:
        graph: networkx Graph
        subgraph_nodes: iterable of nodes in graph
        both_dir (boolean): Whether to include reverse edge from graph.

    Return:
        list of subgraph edges as tuples
    """
    subgraph = graph.subgraph(subgraph_nodes)
    edges = list(subgraph.edges)
    if both_dir:
        reverse_edges = [(edge[1], edge[0]) for edge in edges]
        for edge in reverse_edges:
            if edge not in edges:
                edges.append(edge)
    return edges

def split_index_train_val_test(
    index,
    y,
    train_p=0.8, 
    seed=None
):
    """
    Split a list of indices into training, validation, and test set.

    Args:
        index (list): Indices to split.
        y (list): Corresponding labels for stratification.
        train_p (float): Proportion of training data.
        seed (int): random seed for splitting the data.

    Return:
        index_train, index_val, index_test (list): Indices in each set.
        y_train, y_val, y_test (list): Labels in each set.
    """
    index_train, index_val_test, y_train, y_val_test = train_test_split(
        index, 
        y, 
        train_size=train_p, 
        random_state=seed, 
        stratify=y
    )
    index_val, index_test, y_val, y_test = train_test_split(
        index_val_test, 
        y_val_test, 
        test_size=0.5, 
        random_state=seed, 
        stratify=y_val_test
    )
    return index_train, index_val, index_test, \
        y_train, y_val, y_test

def split_graph_train_val_test(
    graph, 
    roles,
    train_p=0.8, 
    seed=None, 
    both_dir=True
):
    """
    Split a graph into training, validation, and test set. Nodes and edges are split.

    Args:
        graph: networkx Graph
        roles: list of node roles as y
        train_p: float between (0, 1) for proportion of training nodes
        seed: random seed for splitting the data
        both_dir (boolean): Whether to include reverse edge from graph.

    Return:
        node_train, node_val, node_test: lists of nodes for the corresponding data splits
        edge_train, edge_val, edge_test: lists of edges for the coreesponding data splits
        y_train, y_val, y_test: lists of y for the corresponding data splits
    """
    node_all = list(graph.nodes)
    node_train, node_val_test, y_train, y_val_test = train_test_split(
        node_all, 
        roles, 
        train_size=train_p, 
        random_state=seed, 
        stratify=roles
    )

    node_val, node_test, y_val, y_test = train_test_split(
        node_val_test, 
        y_val_test, 
        test_size=0.5, 
        random_state=seed, 
        stratify=y_val_test
    )

    edge_train = get_subgraph_edges(
        graph, 
        node_train, 
        both_dir
    )
    edge_val = get_subgraph_edges(
        graph, 
        node_train + node_val, 
        both_dir
    )
    edge_test = get_subgraph_edges(
        graph, 
        node_all, 
        both_dir
    )
    return node_train, node_val, node_test, \
        edge_train, edge_val, edge_test, \
        y_train, y_val, y_test

def extract_important_subgraph(
    node_indices,
    edge_index,
    edge_score,
    min_subgraph_size
):
    """
    Extract the important subgraph based on edge importance scores and a minimum subgraph
        size constraint.

    Args:
        node_indices (list of int): The nodes required to be in the important subgraph.
        edge_index (torch.long): Edge COO for the original graph with shape 
            [2, num_edges].
        edge_score (torch.float): Edge importance score with shape [num_edges].
        min_subgraph_size (int): Minimum number of nodes in the important subgraph.

    Return:
        important_edge_mask (torch.bool): Important subgraph edge mask.
        important_node_index (torch.long): Node index in the important subgraph.
        size (int): Number of nodes in the important subgraph.
        threshold (float): The edge importance score threshold used to extract the 
            important subgraph.
    """
    thresholds = edge_score.detach().cpu().numpy()
    thresholds = np.flip(
        np.sort(
            np.unique(thresholds)
        )
    )
    for threshold in thresholds:
        next_threshold = True
        subgraph_edge_index = edge_index[:, edge_score >= threshold]
        subgraph = nx.Graph()
        subgraph.add_edges_from(pyg_edge_index_to_tuples(subgraph_edge_index))
        subgraph_comps = [c for c in sorted(nx.connected_components(subgraph), key=len)]
        for comp in subgraph_comps:
            if len(comp) >= min_subgraph_size and set(node_indices).issubset(comp):
                next_threshold = False
                break
        if not next_threshold:
            break
    important_edge_index, _ = torch_geometric.utils.subgraph(
        list(comp),
        subgraph_edge_index
    )
    edge_tuples = pyg_edge_index_to_tuples(edge_index)
    important_edge_tuples = pyg_edge_index_to_tuples(important_edge_index)
    important_edge_mask = [
        edge_tuple in important_edge_tuples for edge_tuple in edge_tuples
    ]
    important_edge_mask = torch.BoolTensor(important_edge_mask)
    important_node_index = torch.LongTensor(list(comp))
    return important_edge_mask, \
        important_node_index, \
        important_node_index.shape[0], \
        threshold

def plot_important_subgraph(
    edge_index,
    important_edge_mask,
    node_color,
    overlay=True,
    G=None,
    pos=None,
    important_node_index=None,
    important_edge_color='#ff0000',
    unimportant_edge_color='#d3d3d3',
    layout=None,
    **kwargs
):
    """
    Plot an important subgraph.

    Args:
        edge_index (torch.long): Edge COO for the original graph with shape 
            [2, num_edges].
        important_edge_mask (torch.bool): Important subgraph edge mask.
        node_color (list of str): Node color hex with len num_nodes.
        overlay (boolean): Whether to overlay on existing graph.
        G (nx.Graph): The existing graph for overlay.
        pos (dict): The node positions for overlay.
        important_node_index (torch.long): Node index in the important subgraph.
        important_edge_color (str): Hex for important edges.
        unimportant_edge_color (str): Hex for unimportant edges.
        layout (nx.drawing.layout): Layout function when overlay is False.
    """
    if overlay:
        assert G is not None, 'G is None when overlay is True'
        assert pos is not None, 'pos is None when overlay is True'
        imp_edges = []
        unimp_edges = []
        for i in range(edge_index.shape[1]):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if important_edge_mask[i]:
                imp_edges.append(edge)
            else:
                unimp_edges.append(edge)
        nx.draw_networkx_nodes(
            G, 
            pos,
            node_color=node_color,
            **kwargs
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=imp_edges,
            edge_color=important_edge_color,
            **kwargs
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=unimp_edges,
            edge_color=unimportant_edge_color,
            alpha=0.5,
            **kwargs
        )
    else:
        assert important_node_index is not None, \
            'important_node_index is None when overlay is False'
        assert layout is not None, 'layout is None when overlay is False'
        sub_edge_index, _ = pyg_subgraph(
            important_node_index,
            edge_index[:, important_edge_mask],
            relabel_nodes=True
        )
        sub_node_color = [node_color[i] for i in important_node_index]
        sub_data = Data(
            edge_index=sub_edge_index, 
            num_nodes=len(sub_node_color)
        )
        sub_G = to_networkx(sub_data)
        sub_pos = layout(sub_G)
        nx.draw_networkx(
            sub_G,
            pos=sub_pos,
            node_color=sub_node_color,
            edge_color=important_edge_color,
            **kwargs
        )

class NormalizedDegree(object):
    def __init__(
        self, 
        mean, 
        std
    ):
        """
        Normalize the node degrees as node features. Code from 
        https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py

        Args:
            mean (float): Mean for normalization.
            std (float): Standard deviation for normalization.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Instance of graph data.
        
        Return:
            data (torch_geometric.data.Data): The original data with x holding normalized
                node degrees.
        """
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def get_dataset(
    name, 
    data_dir='data', 
    sparse=True, 
    cleaned=False
):  
    """
    Get graph datasets from the TU Dortmund University (https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).
    Code adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.

    Args:
        name (str): Dataset name.
        data_dir (str): Data directory to save the downloaded and preprocessed files.
        sparse (boolean): Whether to return in sparse edge COO format.
        cleaned (boolean): Whether to return only non0siomorphic graphs.

    Return:
        dataset (torch_geometric.datasets.tu_dataset.TUDataset): Dataset instance.
    """
    path = osp.join(data_dir, name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])
    return dataset

def calc_dataset_dist(dataset):
    """
    Calculate distributions of the graphs in a dataset.

    Args:
        dataset (torch_geometric.data.Dataset)

    Return:
        results (dict): Result names and values.
    """
    loader = DataLoader(dataset, 1, shuffle=False)
    num_graphs = 0
    num_nodes = []
    num_node_feats = []
    num_edges = []
    y = []
    for data in loader:
        num_nodes.append(data.x.shape[0])
        num_node_feats.append(data.x.shape[1])
        num_edges.append(data.edge_index.shape[1])
        y.append(data.y[0].item())
        num_graphs += 1
    num_nodes = np.array(num_nodes)
    num_node_feats = np.array(num_node_feats)
    num_edges = np.array(num_edges)
    y = np.array(y)
    results = {
        'num_graphs': num_graphs,
        'num_nodes_mean': np.mean(num_nodes),
        'num_nodes_std': np.std(num_nodes),
        'num_node_feats_mean': np.mean(num_node_feats),
        'num_node_feats_std': np.std(num_node_feats),
        'num_edges_mean': np.mean(num_edges),
        'num_edges_std': np.std(num_edges)
    }
    for label in np.unique(y):
        num_label = np.sum(y == label)
        results['num_label_%d' % label] = num_label
        results['prop_label_%d' % label] = num_label / num_graphs
    return results