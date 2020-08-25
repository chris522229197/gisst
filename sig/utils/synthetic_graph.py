# Utilities for generating synthetic graphs

import math
import numpy as np
import networkx as nx


def build_cycle(
    start, 
    num_nodes, 
    role_start=0
):
    """
    Build an undirected cycle graph.

    Args:
        start: starting node index for the graph
        num_nodes: number of nodes in the cycle
        role_start: starting id for the roles

    Return:
        cycle: networkx undirected Graph for the cycle
        roles: list of roles for the cycle nodes
    """
    cycle = nx.Graph()
    cycle.add_nodes_from(range(start, start + num_nodes))
    edges = [(start + i, start + i + 1) for i in range(num_nodes - 1)]
    edges += [(start + num_nodes - 1, start)]
    cycle.add_edges_from(edges)
    roles = [role_start] * num_nodes
    return cycle, roles

def build_grid(
    start, 
    dim, 
    role_start=0
):
    """ 
    Build an undirected grid graph.

    Args:
        start: starting node index for the graph
        dim: number of nodes on each side
        role_start: starting id for the roles
    
    Return:
        grid: networkx undirected Graph for the dim x dim grid
        roles: list of roles for the grid nodes
    """
    grid = nx.grid_graph([dim, dim])
    grid = nx.convert_node_labels_to_integers(grid, first_label=start)
    roles = [role_start] * grid.number_of_nodes()
    return grid, roles

def build_house(
    start, 
    role_start=0
):
    """
    Build a 5-node undirected house-like graph.

    Args:
        start: starting node index for the graph
        role_start: starting id for the roles
    
    Return:
        house: networkx undirected Graph for the house
        roles: list of roles for the house nodes; three roles for the bottom, middle, and
               top of the house
    """
    house = nx.Graph()
    house.add_nodes_from(range(start, start + 5))
    edges = [
        (start, start + 1), # bottom <-> bottom
        (start, start + 3), (start + 1, start + 2), # bottom <-> middle
        (start + 2, start + 3), # middle <-> middle
        (start + 3, start + 4), (start + 2, start + 4) # middle <-> top
    ]
    house.add_edges_from(edges)

    role_middle = role_start + 1
    role_top = role_start + 2
    roles = [
        role_start, role_start, # bottom
        role_middle, role_middle, # middle
        role_top # top
    ]
    return house, roles

def build_tree(
    start, 
    h, 
    r=2, 
    role_start=0
):
    """
    Build a balanced r-tree of height h.

    Args:
        start: starting node index for the graph
        h: height of the tree
        r: number of branches per node
        role_start: starting id for the roles
    
    Return:
        tree: networkx undirected Graph for the tree
        roles: list of roles for the tree nodes
    """
    tree = nx.balanced_tree(r, h)
    tree = nx.convert_node_labels_to_integers(tree, first_label=start)
    roles = [role_start] * tree.number_of_nodes()
    return tree, roles

def build_ba(
    start, 
    n, 
    m=5, 
    role_start=0, 
    seed=None
):
    """
    Build a Barabasi-Albert preferential attachment graph.

    Args:
        start: starting node index for the graph
        n: number of nodes
        m: number of edges to attach from a new node to existing nodes
        role_start: starting id for the roles
        seed: seed for the random Barabasi-Albert model
    
    Return:
        ba: networkx undirected Graph for the Barabasi-Albert graph
        roles: list of roles for the nodes
    """
    ba = nx.barabasi_albert_graph(n, m, seed)
    ba = nx.convert_node_labels_to_integers(ba, first_label=start)
    roles = [role_start] * n
    return ba, roles

def build_base_motif(
    base_type,
    base_kwargs,
    list_motifs,
    start=0,
    role_start=0,
    rand_plugin=True,
    rand_seed=None
):
    """
    Build a base graph with attached motifs.
    
    Args:
        base_type: str for the base type (cycle, grid, house, tree, or ba)
        base_kwargs: kwargs for for building the base graph, excluding start and
                     role_start
        list_motifs: list of motif lists (1st item: str for mofit type; 2nd item: kwargs
                     excluding start and role_start)
        start: starting node index for the entire graph
        role_start: starting id for the roles in the entire graph
        rand_plugin: boolean for whether the mofits are randomly plugged into the base
        rand_seed: seed for random plugins

    Return:
        base: a networkx Graph with the base and attached motifs
        roles: list of role labels for all the nodes
        plugins: list of node indices where the motifs are attached
    """
    # Build base
    base, roles = eval('build_' + base_type)(
        start=start,
        role_start=role_start, 
        **base_kwargs
    )

    base_size = base.number_of_nodes()
    base_nodes = np.array(base.nodes)
    nx.set_edge_attributes( # base edge importance for base nodes
        base,
        list(base.nodes),
        'important_for'
    )
    motif_count = len(list_motifs)

    role_start_lookup = {('base', base_size): role_start}
    start += base_size # increment the starting node index for the motifs

    # Find plugin node indices
    if rand_plugin:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        plugins = np.random.choice(
            base_nodes, 
            motif_count, 
            replace=False
        ).tolist()
    else:
        plugins = np.linspace(
            np.min(base_nodes), 
            np.max(base_nodes), 
            motif_count
        )
        plugins = [int(k) for k in plugins]
    plugins.sort()

    # Build motifs and attach them to the base
    for motif_id, motif in enumerate(list_motifs):
        motif_type = motif[0]
        motif_kwargs = {
            **{'start': start, 'role_start': 0}, 
            **motif[1]
        }
        motif_graph, motif_roles = eval('build_' + motif_type)(**motif_kwargs)
        motif_size = motif_graph.number_of_nodes()
        motif_key = (motif_type, motif_size)
        nx.set_edge_attributes( # motif edge importance for motif nodes
            motif_graph,
            list(motif_graph.nodes), 
            'important_for'
        )
        
        if motif_key in role_start_lookup.keys():
            motif_role_start = role_start_lookup[motif_key]
        else:
            motif_role_start = max(roles) + 1
            role_start_lookup[motif_key] = motif_role_start
        motif_roles = [role + motif_role_start for role in motif_roles]

        base.add_nodes_from(motif_graph.nodes())
        base.add_edges_from(motif_graph.edges(data=True))
        base.add_edges_from([(start, plugins[motif_id])])

        roles += motif_roles
        start += motif_size
    return base, roles, plugins

def add_rand_edges(
    graph, 
    p,
    seed=None
):
    """
    Add random edges to a graph.

    Args:
        graph: networkx Graph
        p: proportion of edges to add based on number of edges in graph
        seed: seed for numpy random

    Return:
        rand_graph: copy of graph with added random edges
    """
    if seed is not None:
        np.random.seed(seed)

    rand_graph = graph.copy()
    nodes = np.array(rand_graph.nodes)
    edge_count = int(rand_graph.number_of_edges() * p)
    for _ in range(edge_count):
        while True:
            source = np.random.choice(nodes)
            target = np.random.choice(nodes[nodes != source])
            if not rand_graph.has_edge(source, target):
                break
        rand_graph.add_edge(source, target)
    return rand_graph

def rand_join_graphs(
    graph_1, 
    graph_2, 
    num_rand_edges=0, 
    seed=None
):
    """
    Join two graphs with random edges between the graphs.

    Args:
        graph_1, graph_2: networkx Graphs to join
        num_rand_edges: number of random edges between graph_1 and graph_2; 0 for no 
                        random edge connections
        seed: seed for numpy random

    Return:
        joined_graph: graph_1 and graph_2 joined with random edge connections
    """
    if seed is not None:
        np.random.seed(seed)

    joined_graph = nx.compose(graph_1, graph_2)
    edge_count = 0
    while edge_count <= num_rand_edges:
        source = np.random.choice(graph_1.nodes())
        target = np.random.choice(graph_2.nodes())
        if source != target:
            joined_graph.add_edge(source, target)
            edge_count += 1
    return joined_graph

def gen_node_feat(
    roles, 
    num_feat, 
    mu_scale=1, 
    sigma_scale=0.2, 
    seed=None
):
    """
    Generate independent Gaussian node features based on node roles.

    Args:
        roles: list of node roles; same size as number of nodes
        num_feat: number of independent Gaussian features
        mu_scale: the scale of Gaussian mean with respect to the standard deviation
        sigma_scale: the scale of Gaussian standard deviation with respect to the mean
        seed: seed for numpy random

    Return:
        node_feat: numpy array of shape num_nodes x num_feat
    """
    if seed is not None:
        np.random.seed(seed)

    roles = np.array(roles)
    uniq_roles = list(set(roles))
    uniq_roles.sort()
    num_roles = len(uniq_roles)

    mu_max = num_roles / 2
    mu_min = -mu_max
    mus = np.arange(mu_min, mu_max, 1) * mu_scale

    node_feat = np.zeros((roles.shape[0], num_feat))
    for i in range(num_roles):
        role = uniq_roles[i]
        role_mask = roles == role
        node_feat[role_mask, :] = np.random.normal(
            mus[i], 
            sigma_scale,
            (np.sum(role_mask), num_feat)
        )
    return node_feat

def add_rand_node_feat(
    node_feat, 
    p,
    mu=0,
    sigma=1,
    seed=None
):
    """
    Concatenate random noisy Gaussian node features regardless of node roles.

    Args:
        node_feat: numpy array of node features
        p: proportion of random features to add based on existing number of features
        mu: Gaussian mean of the random features
        sigma: Gaussian standard deviation of the random features
        seed: seed for numpy random
    
    Return:
        numpy array of node_feat appended with random features
    """
    if seed is not None:
        np.random.seed(seed)

    num_rand_feat = int(node_feat.shape[1] * p)
    rand_node_feat = np.random.normal(
        mu,
        sigma,
        (node_feat.shape[0], num_rand_feat)
    )
    return np.concatenate((node_feat, rand_node_feat), axis=1)