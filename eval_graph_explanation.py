# Quantitatively evaluate graph-level explanation

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import networkx as nx
from sklearn.metrics import accuracy_score
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

def write_eval_statistics(
    y,
    pred,
    occluded_pred,
    node_sparsity,
    edge_sparsity,
    outfile,
    prefix=''
):
    """
    Write the evaluation statistics for graph-level explanations.

    Args:
        y (np.array): Ground truth classification label.
        pred (np.array): Predicted label based on original data.
        occluded_pred (np.array): Predicted label after masking the node features in the
            important subgraph.
        node_sparsity (np.array): 1 - subgraph_num_nodes / graph_num_nodes
        edge_sparsity (np.array): 1 - subgraph_num_edges / graph_num_edges
        outfile (file object): Output file.
        prefix (str): Prefix for each metric name.

    Return:
        Write the statistics to a file.
    """
    fidelity = accuracy_score(y, pred) - accuracy_score(y, occluded_pred)
    outfile.write(
        '%sfidelity: %0.3f\n' % (
            prefix, 
            fidelity
        )
    )
    outfile.write(
        '%snode_sparsity: %0.3f +/- %0.3f\n' % (
            prefix, 
            np.mean(node_sparsity),
            np.std(node_sparsity)
        )
    )
    outfile.write(
        '%sedge_sparsity: %0.3f +/- %0.3f\n' % (
            prefix, 
            np.mean(edge_sparsity),
            np.std(edge_sparsity)
        )
    )

def main():
    parser = argparse.ArgumentParser(
        description="evaluate the explanation of a trained GNN graph classifier"
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=[
            'mutagenicity', 
            'reddit_binary',
            'proteins'
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
        default=15,
        help='minimum number of nodes for extracted important subgraph',
        dest='min_subgraph_size'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random seed',
        dest='seed'
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

    args = parser.parse_args()
    assert args.model_type in allowed_models[args.explainer_type], \
        '{} cannot be used for model_type={}'.format(args.explainer_type, args.model_type)

    print('Running {} with arguments'.format(sys.argv[0]))
    for arg in vars(args):
        print('\t{}={}'.format(arg, getattr(args, arg)))
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict_file = args.model_dir + 'best_state_dict.pt'
    hyperparam_file = args.model_dir + 'best_hyperparams.txt'
    output_file = args.model_dir + args.explainer_type + '_explainer_eval_results'
    log_file = args.model_dir + args.explainer_type + '_explainer_eval_log'

    if args.ablation != '':
        output_file += '_abl_{}'.format(args.ablation)
        log_file += '_abl_{}'.format(args.ablation)

    output_file += '_minsize_{}.txt'.format(args.min_subgraph_size)
    log_file += '_minsize_{}.txt'.format(args.min_subgraph_size)

    io_utils.print_log('seed={}'.format(args.seed), log_file=log_file, mode='w')
    io_utils.print_log('device={}'.format(device), log_file=log_file)

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
    explainer.to(device)

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
    
    y_lst = []
    pred_lst = []
    occluded_pred_lst = []
    node_sparsity_lst = []
    edge_sparsity_lst = []

    counter = 0
    for data_test in loader_test:
        data_test = data_test.to(device)
        label = data_test.y.item()
        out = model(
            data_test.x,
            data_test.edge_index,
            batch=data_test.batch
        )
        pred = out.max(1)[1].item()

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

        node_sparsity = 1 - important_node_index.shape[0] / data_test.x.shape[0]
        edge_sparsity = 1 - important_edge_mask.sum().item() / data_test.edge_index.shape[1]
        occluded_data = data_test.clone()
        occluded_data.x[important_node_index, :] = 0
        occluded_out = model(
            occluded_data.x,
            occluded_data.edge_index,
            batch=occluded_data.batch
        )
        occluded_pred = occluded_out.max(1)[1].item()
        
        y_lst.append(label)
        pred_lst.append(pred)
        occluded_pred_lst.append(occluded_pred)
        node_sparsity_lst.append(node_sparsity)
        edge_sparsity_lst.append(edge_sparsity)

        counter += 1
        io_utils.print_log(
            'Finished explaining {} graphs'.format(counter),
            log_file=log_file
        )
        if args.debug and counter == 5:
            break

    with open(output_file, 'w') as txtfile:
        txtfile.write('test dataset all graphs\n')
        write_eval_statistics(
            np.array(y_lst),
            np.array(pred_lst),
            np.array(occluded_pred_lst),
            np.array(node_sparsity_lst),
            np.array(edge_sparsity_lst),
            txtfile,
            prefix='\t'
        )
    io_utils.print_log('Done!', log_file=log_file)

if __name__ == '__main__':
    main()