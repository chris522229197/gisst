# Utilities for PyTorch Geometric (PyG)

import numpy as np
import torch
from torch_geometric.data import Data


def get_mask(
    ref, 
    sub, 
    pyg=False
):
    """
    Get the mask for a reference list based on a subset list.

    Args:
        ref: reference list
        sub: subset list
        pyg: boolean; whether to return torch tensor for PyG

    Return:
        mask: list or torch.BoolTensor
    """
    mask = [item in sub for item in ref]
    if pyg:
        mask = torch.BoolTensor(mask)
    return mask

def get_pyg_edge_index(graph, both_dir=True):
    """
    Get edge_index for an instance of torch_geometric.data.Data.

    Args:
        graph: networkx Graph
        both_dir: boolean; whether to include reverse edge from graph
    
    Return:
        torch.LongTensor of edge_index with size [2, num_edges]
    """
    sources = []
    targets = []
    for edge in graph.edges:
        sources.append(edge[0])
        targets.append(edge[1])
        if both_dir:
            sources.append(edge[1])
            targets.append(edge[0])
    return torch.LongTensor([sources, targets])

def pyg_edge_index_to_tuples(edge_index):
    """
    Convert edge_index to a list of (source, target) tuples.

    Args:
        edge_index (torch.long): Edge COO with shape [2, num_edges].
    
    Return:
        edge_tuples (list of tuple): Edge tuples.
    """
    edge_tuples = [
        (edge_index[0, i].item(), edge_index[1, i].item()) \
            for i in range(edge_index.shape[1])
    ]
    return edge_tuples

def get_pyg_imp_edge_mask(
    graph, 
    edge_index, 
    imp_edge_attr='important_for'
):
    """
    Get a tensor for node-specific edge importance.

    Args:
        graph: networkx Graph
        edge_index: torch.LongTensor for COO edges with size [2, num_edges]
        imp_edge_attr: the edge attribute in graph storing the nodes each edge is 
                       important for

    Return:
        imp_edge_mask: torch.BoolTensor of size [num_nodes, num_edges]
    """
    imp_edge_mask = []
    for i in range(edge_index.shape[1]):
        edge = (edge_index[0, i].item(), edge_index[1, i].item())
        reverse_edge = edge[::-1]
        
        if edge in graph.edges and imp_edge_attr in graph.edges[edge]:
            imp_edge_mask.append(
                get_mask(graph.nodes, graph.edges[edge][imp_edge_attr])
            )
        elif reverse_edge in graph.edges and imp_edge_attr in graph.edges[reverse_edge]:
            imp_edge_mask.append(
                get_mask(graph.nodes, graph.edges[reverse_edge][imp_edge_attr])
            )
        else:
            imp_edge_mask.append(
                torch.BoolTensor([False] * graph.number_of_nodes())
            )
            
    imp_edge_mask = torch.BoolTensor(imp_edge_mask).permute(1, 0)
    return imp_edge_mask

def process_pyg_data(
    data, 
    graph_nodes, 
    graph_edges,
    subgraph_nodes, 
    subgraph_edges, 
    subgraph_name
): 
    """
    Add node_mask and edge_mask for a subgraph to torch_geometric.data.Data

    Args:
        data: an instance of torch_geometric.data.Data
        graph_nodes: iterable of all the nodes in the graph
        graph_edges: iterable of all edge tuples in the graph
        subgraph_nodes: list of subgraph nodes
        subgraph_edges: list of subgraph edge tuples
        subgraph_name: str for the subgraph name
    
    Return:
        data (torch_geometric.data.Data): Data with node_mask and edge_mask.
    """
    data['node_mask_' + subgraph_name] = get_mask(graph_nodes, subgraph_nodes, pyg=True)
    data['edge_mask_' + subgraph_name] = get_mask(graph_edges, subgraph_edges, pyg=True)
    return data