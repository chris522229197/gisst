# Explain the node feature and edge importance of a trained Graph Neural Network (GNN)
# graph classifier.

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, subgraph

from sig.utils import io_utils
from sig.utils.graph_utils import extract_important_subgraph, plot_important_subgraph
from sig.nn.models.gcn import GCN
from sig.nn.models.gat import GAT
from sig.nn.models.sigcn import SIGCN
from sig.explainers.grad_explainer import GradExplainer
from sig.explainers.gat_explainer import GATExplainer
from sig.explainers.sig_explainer import SIGExplainer
from sig.explainers.gnn_explainer import GNNExplainer
from sig.nn.loss.classification_loss import cross_entropy_loss


def parse_txt_hyperparams(txtfile):
    """
    Parse hyperparameters from a txt file.
    
    Args:
        txtfile (str): Path for the hyperparameter txt file.
        
    Return:
        hyperparams (dict): Hyperparameter names and values.
    """
    dtypes = {
        'num_hidden_layer': int,
        'hidden_dim': int,
        'learning_rate': float,
        'num_epochs': int,
        'l2_coeff': float,
        'dropout_rate': float,
        'x_l1_coeff': float,
        'x_ent_coeff': float,
        'edge_l1_coeff': float,
        'edge_ent_coeff': float
    }
    hyperparams = {}
    with open(txtfile, 'r') as f:
        for line in f:
            parsed_line = line.split(': ')
            name = parsed_line[0]
            value = parsed_line[1].replace('\n', '')
            if value == 'None':
                value = None
            else:
                value = dtypes[name](value)
            hyperparams[name] = value
    return hyperparams

def load_model(
    model_type,
    dataset,
    hyperparams,
    state_dict_file,
    eval_mode=True,
    **kwargs
):
    """
    Load a trained GNN graph classifier.
    
    Args:
        model_type (str): Model type {'gcn', 'gat', 'sigcn'}.
        data (torch_geometric.data.Data): Graph data.
        hyperparams (dict): Hyperparameter names and values.
        state_dict_file (str): Path to the PyTorch model state dict.
        eval_mode (boolean): Whether to put the model in eval mode.
    
    Return:
        model (torch.nn.Module): Loaded trained GNN graph classifier.
    """
    loader = DataLoader(
        dataset[dataset.index_train],
        1,
        shuffle=False,
        num_workers=0
    )
    for data in loader:
        input_size = data.x.shape[1]
        break
    num_classes = len(set(dataset.data.y.tolist()))
    num_hidden_layer = hyperparams['num_hidden_layer']
    hidden_dim = hyperparams['hidden_dim']
    dropout_rate = hyperparams['dropout_rate']
    if eval_mode:
        dropout_rate = 0
    
    model = eval(model_type.upper())(
        input_size=input_size,
        output_size=num_classes,
        hidden_conv_sizes=(hidden_dim, ) * num_hidden_layer,
        hidden_dropout_probs=(dropout_rate, ) * num_hidden_layer,
        classify_graph=True,
        lin_dropout_prob=dropout_rate
    )
    model.load_state_dict(torch.load(state_dict_file, **kwargs))
    if eval_mode:
        model.eval()
    return model

def mutagenicity_node_color(data):
    """
    Get the node color for a graph in Mutagenicity.

    Args:
        data (torch_geometric.data.Data): Data for a graph.

    Return:
        list of node color in hex str
    """
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    cmap = sns.color_palette("muted").as_hex() + flatui
    node_type = data.x.nonzero()[:, 1]
    return [cmap[i] for i in node_type], [i.item() for i in node_type]

def reddit_binary_node_color(data):
    """
    Get the node color for a graph in REDDIT-BINARY.

    Args:
        data (torch_geometric.data.Data): Data for a graph.
    
    Return:
        list of node color in hex str
    """
    num_nodes = data.x.shape[0]
    color = sns.color_palette("muted").as_hex()[0]
    return [color] * num_nodes, [0] * num_nodes

def main():
    parser = argparse.ArgumentParser(
        description="explain a trained GNN graph classifier"
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=[
            'mutagenicity', 
            'reddit_binary'
        ],
        help='dataset to use'
    )
    parser.add_argument(
        'model_dir',
        type=str,
        help='directory containing model files'
    )
    parser.add_argument(
        'model_type',
        type=str,
        choices=[
            'gcn',
            'gat',
            'sigcn'
        ],
        help='type of trained GNN model to explain'
    )
    parser.add_argument(
        'explainer_type',
        type=str,
        choices=[
            'grad',
            'abs_grad',
            'mag_grad',
            'pred_grad',
            'mag_pred_grad',
            'gat',
            'sig',
            'sig_grad',
            'abs_sig_grad',
            'pred_sig_grad',
            'gnn'
        ],
        help='type of explainer to use'
    )
    parser.add_argument(
        '--min-subgraph-size',
        type=int,
        default=10,
        help='minimum number of nodes for extracted important subgraph',
        dest='min_subgraph_size'
    )
    parser.add_argument(
        '--plot-mode',
        type=str,
        default='graph',
        choices=[
            'graph',
            'node_feat'
        ],
        help='plot mode',
        dest='plot_mode'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random seed',
        dest='seed'
    )
    parser.add_argument(
        '--layout',
        type=str,
        default='kamada_kawai',
        choices=[
            'kamada_kawai',
            'spring'
        ],
        help='layout for graph visualization',
        dest='layout'
    )
    parser.add_argument(
        '--num-graphs-per-page',
        type=int,
        default=5,
        help='number of graphs to visualize per page in the pdf output',
        dest='num_graphs_per_page'
    )
    parser.add_argument(
        '--max-graph-size',
        type=int,
        default=300,
        help='maximum graph size for visualization',
        dest='max_graph_size'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        default='',
        choices=[
            '',
            'edge',
            'ent',
            'l1',
            'x'
        ],
        help='regularization ablation',
        dest='ablation'
    )
    parser.add_argument(
        '--debug',
        help='debug mode to run through 5 graphs',
        dest='debug',
        action='store_true'
    )
    parser.set_defaults(debug=False)

    allowed_models = {
        'grad': ['gcn', 'gat', 'sigcn'],
        'abs_grad': ['gcn', 'gat', 'sigcn'],
        'mag_grad': ['gcn', 'gat', 'sigcn'],
        'pred_grad': ['gcn', 'gat', 'sigcn'],
        'mag_pred_grad': ['gcn', 'gat', 'sigcn'],
        'gat': ['gat'],
        'sig': ['sigcn'],
        'sig_grad': ['sigcn'],
        'abs_sig_grad': ['sigcn'],
        'pred_sig_grad': ['sigcn'],
        'gnn': ['gcn', 'gat', 'sigcn']
    }
    allowed_plot_mode = {
        'grad': ['graph', 'node_feat'],
        'abs_grad': ['graph', 'node_feat'],
        'mag_grad': ['graph', 'node_feat'],
        'pred_grad': ['graph', 'node_feat'],
        'mag_pred_grad': ['graph', 'node_feat'],
        'gat': ['graph'],
        'sig': ['graph', 'node_feat'],
        'sig_grad': ['graph', 'node_feat'],
        'abs_sig_grad': ['graph', 'node_feat'],
        'pred_sig_grad': ['graph', 'node_feat'],
        'gnn': ['graph', 'node_feat']
    }
    explainer_class_lookup = {
        'grad': GradExplainer,
        'abs_grad': GradExplainer,
        'mag_grad': GradExplainer,
        'pred_grad': GradExplainer,
        'mag_pred_grad': GradExplainer,
        'gat': GATExplainer,
        'sig': SIGExplainer,
        'sig_grad': SIGExplainer,
        'abs_sig_grad': SIGExplainer,
        'pred_sig_grad': SIGExplainer,
        'gnn': GNNExplainer
    }
    explainer_init_kwargs = {
        'grad': {},
        'abs_grad': {},
        'mag_grad': {},
        'pred_grad': {},
        'mag_pred_grad': {},
        'gat': {},
        'sig': {},
        'sig_grad': {},
        'abs_sig_grad': {},
        'pred_sig_grad': {},
        'gnn': {
            'epochs': 100, 
            'log': False
        }
    }
    layout_lookup = {
        'kamada_kawai': nx.kamada_kawai_layout,
        'spring':  nx.spring_layout
    }
    node_color_fn_lookup = {
        'mutagenicity': mutagenicity_node_color, 
        'reddit_binary': reddit_binary_node_color
    }

    args = parser.parse_args()
    assert args.model_type in allowed_models[args.explainer_type], \
        '{} cannot be used for model_type={}'.format(args.explainer_type, args.model_type)
    assert args.plot_mode in allowed_plot_mode[args.explainer_type], \
        '{} cannot be used for plot_mode={}'.format(args.explainer_type, args.plot_mode)

    print('Running {} with arguments'.format(sys.argv[0]))
    for arg in vars(args):
        print('\t{}={}'.format(arg, getattr(args, arg)))

    draw_kwargs = {}
    draw_kwargs['width'] = 5
    draw_kwargs['arrows'] = False
    draw_kwargs['with_labels'] = False
    draw_kwargs['node_size'] = 450
    figsize = 45
    if args.plot_mode == 'graph':
        ncols = 3
    else:
        ncols = 2
    def new_subplots():
        return plt.subplots(
            nrows=args.num_graphs_per_page, 
            ncols=ncols, 
            figsize=(figsize, figsize)
        )
    layout = layout_lookup[args.layout]
    node_color_fn = node_color_fn_lookup[args.dataset]
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_file = args.model_dir + 'best_state_dict.pt'
    hyperparam_file = args.model_dir + 'best_hyperparams.txt'
    output_file = args.model_dir + args.explainer_type + \
        '_explainer_{}_results'.format(args.plot_mode)
    log_file = args.model_dir + args.explainer_type + \
        '_explainer_{}_log'.format(args.plot_mode)
    object_subdir = args.model_dir + args.explainer_type + \
        '_explainer_{}_files'.format(args.plot_mode)

    if args.ablation != '':
        output_file += '_abl_{}'.format(args.ablation)
        log_file += '_abl_{}'.format(args.ablation)
        object_subdir += '_abl_{}'.format(args.ablation)

    if args.plot_mode == 'graph':
        output_file += '_minsize_{}.pdf'.format(args.min_subgraph_size)
        log_file += '_minsize_{}.txt'.format(args.min_subgraph_size)
        object_subdir += '_minsize_{}/'.format(args.min_subgraph_size)
    else:
        output_file += '.pdf'
        log_file += '.txt'
        object_subdir += '/'

    io_utils.print_log('seed={}'.format(args.seed), log_file=log_file, mode='w')
    io_utils.print_log('device={}'.format(device), log_file=log_file)
    if not os.path.exists(object_subdir):
        os.makedirs(object_subdir)

    dataset = torch.load(
        'graphs/{}/dataset.pyg.pkl'.format(args.dataset)
    )
    hyperparams = parse_txt_hyperparams(hyperparam_file)
    model = load_model(
        args.model_type,
        dataset,
        hyperparams,
        state_dict_file, 
        map_location=device
    )
    explainer = explainer_class_lookup[args.explainer_type](
        model,
        **explainer_init_kwargs[args.explainer_type]
    )

    if args.ablation == 'edge':
        explainer.coeffs = {
            'edge_size': 0,
            'node_feat_size': 1.0,
            'edge_ent': 0,
            'node_feat_ent': 0.1,
        }
    elif args.ablation == 'ent':
        explainer.coeffs = {
            'edge_size': 0.005,
            'node_feat_size': 1.0,
            'edge_ent': 0,
            'node_feat_ent': 0,
        }
    elif args.ablation == 'l1':
        explainer.coeffs = {
            'edge_size': 0,
            'node_feat_size': 0,
            'edge_ent': 1.0,
            'node_feat_ent': 0.1,
        }
    elif args.ablation == 'x':
        explainer.coeffs = {
            'edge_size': 0.005,
            'node_feat_size': 0,
            'edge_ent': 1.0,
            'node_feat_ent': 0,
        }
        
    loader_test = DataLoader(
        dataset[dataset.index_test],
        1,
        shuffle=False,
        num_workers=0
    )

    fig, axes = new_subplots()
    with PdfPages(output_file) as pdf:
        graph_index = 0
        graph_counter = 1
        for data_test in loader_test:
            data_test = data_test.to(device)
            if data_test.x.shape[0] <= args.max_graph_size:
                label = data_test.y.item()
                out = model(
                    data_test.x,
                    data_test.edge_index,
                    batch=data_test.batch
                )
                pred = out.max(1)[1].item()
                if pred == label:
                    pred_result = 'correct'
                else:
                    pred_result = 'incorrect'
                title = 'index: {}'.format(graph_index)
                ax_index = (graph_counter - 1) % args.num_graphs_per_page

                G = to_networkx(data_test)
                pos = layout(G)
                node_color, node_type = node_color_fn(data_test)
                
                ax = axes[ax_index, 0]
                ax.set_title(title + ' original', fontdict={'fontsize': figsize})
                nx.draw_networkx(
                    G,
                    pos=pos,
                    node_color=node_color,
                    ax=ax,
                    **draw_kwargs
                )

                explainer_eval_kwargs = {
                    'grad': {
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss
                    },
                    'abs_grad': {
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss,
                        'take_abs': True
                    },
                    'mag_grad': {
                        'y': data_test.y,
                        'loss_fn': cross_entropy_loss,
                        'take_mag': True 
                    }, 
                    'pred_grad': {
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss,
                        'pred_for_grad': True
                    },
                    'mag_pred_grad': {
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss,
                        'take_mag': True,
                        'pred_for_grad': True
                    },
                    'gat': {},
                    'sig': {},
                    'sig_grad': {
                        'use_grad': True,
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss
                    },
                    'abs_sig_grad': {
                        'use_grad': True,
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss,
                        'take_abs': True
                    },
                    'pred_sig_grad': {
                        'use_grad': True,
                        'y': data_test.y, 
                        'loss_fn': cross_entropy_loss,
                        'pred_for_grad': True
                    },
                    'gnn': {}
                }
                
                node_feat_score, edge_score = explainer.explain_graph(
                    x=data_test.x,
                    edge_index=data_test.edge_index,
                    batch=data_test.batch, 
                    **explainer_eval_kwargs[args.explainer_type]
                )                
                important_edge_mask, important_node_index, _, _ = extract_important_subgraph(
                    [],
                    data_test.edge_index,
                    edge_score,
                    args.min_subgraph_size
                )
                
                subtitle = title + ' {}; result={}; label={}; pred={}'.format(
                    args.explainer_type, 
                    pred_result, 
                    label, 
                    pred
                )
                graph_info = {}
                graph_info['label'] = label
                graph_info['pred'] = pred
                graph_info['pos'] = pos
                graph_info['node_type'] = node_type

                if args.plot_mode == 'graph':
                    ax = axes[ax_index, 1]
                    ax.set_title(
                        subtitle, 
                        fontdict={'fontsize': figsize}
                    )
                    nx.draw_networkx(
                        G,
                        pos=pos,
                        node_color=node_color,
                        ax=ax,
                        edge_color=edge_score.detach().cpu().numpy(),
                        edge_cmap=plt.cm.YlOrBr,
                        **draw_kwargs
                    )

                    ax = axes[ax_index, 2]
                    ax.set_title(
                        title + ' {}'.format(args.explainer_type), 
                        fontdict={'fontsize': figsize}
                    )
                    plot_important_subgraph(
                        data_test.edge_index,
                        important_edge_mask,
                        node_color,
                        G=G,
                        pos=pos,
                        ax=ax,
                        **draw_kwargs
                    )
                    graph_info['edge_score'] = edge_score
                    graph_info['edge_index'] = data_test.edge_index
                    graph_info['important_edge_mask'] = important_edge_mask
                elif args.plot_mode == 'node_feat' and node_feat_score is not None:
                    ax = axes[ax_index, 1]
                    ax.set_title(subtitle, fontdict={'fontsize': figsize})
                    ax.imshow(
                        node_feat_score.detach().cpu().numpy()[np.newaxis, :], 
                        cmap='Reds', 
                        aspect='auto'
                    )
                    ax.set_xticks(
                        np.arange(-.5, node_feat_score.shape[0], 1), 
                        minor=True
                    )
                    ax.grid(
                        which='minor', 
                        color='black', 
                        linestyle='-', 
                        linewidth=2
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    graph_info['node_feat_score'] = node_feat_score
                # save the files for later plotting
                file_prefix = 'index_{}'.format(graph_index)
                nx.write_gpickle(
                    G, 
                    object_subdir + file_prefix + '.gpkl'
                )
                torch.save(
                    graph_info,
                    object_subdir + file_prefix + '_info.pt'
                )

                if graph_counter >= args.num_graphs_per_page and \
                graph_counter % args.num_graphs_per_page == 0:
                    pdf.savefig()
                    plt.close(fig)
                    fig, axes = new_subplots()
                    io_utils.print_log(
                        'Finished explaining {} graphs'.format(graph_counter),
                        log_file=log_file
                    )
                if args.debug and graph_counter == 5:
                    break
                graph_counter += 1
            graph_index += 1
    io_utils.print_log('Done!', log_file=log_file)

if __name__ == '__main__':
    main()