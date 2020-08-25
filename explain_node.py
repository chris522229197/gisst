# Explain the node feature and edge importance of a trained Graph Neural Network (GNN)
# node classifier.

import os
import sys
import argparse
import random
import numpy as np
from functools import partial
import torch
from torch_geometric.nn import GNNExplainer
from sig.utils import io_utils
from sig.nn.models.gcn import GCN
from sig.nn.models.gat import GAT
from sig.nn.models.sigcn import SIGCN
from sig.explainers.grad_explainer import GradExplainer
from sig.explainers.gat_explainer import GATExplainer
from sig.explainers.sig_explainer import SIGExplainer
from sig.nn.loss.classification_loss import cross_entropy_loss
from sig.utils.optimization_utils import evaluate_explanation


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
    data,
    hyperparams,
    state_dict_file,
    eval_mode=True,
    **kwargs
):
    """
    Load a trained model.
    
    Args:
        model_type (str): Model type {'gcn', 'gat', 'sigcn'}.
        data (torch_geometric.data.Data): Graph data.
        hyperparams (dict): Hyperparameter names and values.
        state_dict_file (str): Path to the PyTorch model state dict.
        eval_mode (boolean): Whether to put the model in eval mode.
    
    Return:
        model (torch.nn.Module): Loaded trained model.
    """
    input_size = data.x.shape[1]
    num_classes = len(set(data.y.tolist()))
    num_hidden_layer = hyperparams['num_hidden_layer']
    hidden_dim = hyperparams['hidden_dim']
    dropout_rate = hyperparams['dropout_rate']
    if eval_mode:
        dropout_rate = 0
    
    model = eval(model_type.upper())(
        input_size=input_size,
        output_size=num_classes,
        hidden_conv_sizes=(hidden_dim, ) * num_hidden_layer,
        hidden_dropout_probs=(dropout_rate, ) * num_hidden_layer
    )
    model.load_state_dict(torch.load(state_dict_file, **kwargs))
    if eval_mode:
        model.eval()
    return model

def process_result(result):
    """
    Process result value into str.

    Args:
        result (float or None)

    Return: str
    """
    return 'None' if result is None else '%0.4f' % result

def write_explainer_results(
    results,
    outfile,
    prefix='' 
):
    """
    Write explainer evaluation results to a text file.

    Args:
        results (dict): Performance result names and values.
        edge_auroc (float or None): Edge explanation AUROC.
         outfile (file object): Output file.
        prefix (str): Prefix for each metric name.
    
    Return:
        No objects returned.
    """
    for name, value in results.items():
        outfile.write(
            '{}{}: {}\n'.format(
                prefix, 
                name,
                process_result(value)
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="explain a trained GNN node classifier"
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=[
            'ba_300_house_80', 
            'ba_300_house_80_comm', 
            'tree_8_cycle_80', 
            'tree_8_grid_80',
            'ba_300_cycle_80',
            'ba_300_grid_80',
            'tree_8_house_80'
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
    basis_class_lookup = {
        'ba_300_house_80': [0],
        'ba_300_house_80_comm': [0, 4],
        'tree_8_cycle_80': [0], 
        'tree_8_grid_80': [0],
        'ba_300_cycle_80': [0],
        'ba_300_grid_80': [0],
        'tree_8_house_80': [0]
    }
    min_subgraph_size_lookup = {
        'ba_300_house_80': 5,
        'ba_300_house_80_comm': 5,
        'tree_8_cycle_80': 6, 
        'tree_8_grid_80': 9,
        'ba_300_cycle_80': 6,
        'ba_300_grid_80': 9,
        'tree_8_house_80': 5
    }

    args = parser.parse_args()
    assert args.model_type in allowed_models[args.explainer_type], \
        '{} cannot be used for {}'.format(args.explainer_type, args.model_type)
    
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
    output_file = args.model_dir + args.explainer_type + '_explainer_results'
    log_file = args.model_dir + args.explainer_type + '_explainer_log'
    if args.ablation != '':
        output_file += '_abl_{}'.format(args.ablation)
        log_file += '_abl_{}'.format(args.ablation)
    output_file += '.txt'
    log_file += '.txt'

    io_utils.print_log('seed={}'.format(args.seed), log_file=log_file, mode='w')
    io_utils.print_log('device={}'.format(device), log_file=log_file)

    data = torch.load(
        'graphs/{}/data.pyg.pkl'.format(args.dataset)
    )
    data = data.to(device)
    explainer_eval_kwargs = {
        'grad': {
            'y': data.y, 
            'loss_fn': cross_entropy_loss
        },
        'abs_grad': {
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'take_abs': True
        },
        'mag_grad': {
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'take_mag': True
        },
        'pred_grad': {
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'pred_for_grad': True
        },
        'mag_pred_grad': {
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'pred_for_grad': True,
            'take_mag': True
        },
        'gat': {},
        'sig': {},
        'sig_grad': {
            'use_grad': True,
            'y': data.y, 
            'loss_fn': cross_entropy_loss
        },
        'abs_sig_grad': {
            'use_grad': True,
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'take_abs': True
        },
        'pred_sig_grad': {
            'use_grad': True,
            'y': data.y, 
            'loss_fn': cross_entropy_loss,
            'pred_for_grad': True
        },
        'gnn': {}
    }

    hyperparams = parse_txt_hyperparams(hyperparam_file)
    model = load_model(
        args.model_type,
        data,
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

    node_index = torch.LongTensor(np.arange(data.x.shape[0]))
    node_index_test = node_index[data.node_mask_test].tolist()
    basis_class = basis_class_lookup[args.dataset]
    basis_node_index_test = [i for i in node_index_test if data.y[i] in basis_class]
    motif_node_index_test = [i for i in node_index_test if data.y[i] not in basis_class]

    edge_imp = data.imp_edge_mask.type(torch.LongTensor)
    node_feat_imp = data.imp_node_feat_mask.type(torch.LongTensor)
    overall_node_feat_imp = node_feat_imp[0, :]

    evaluate_explanation_partial = partial(
        evaluate_explanation,
        explainer=explainer,
        x=data.x,
        edge_index=data.edge_index,
        min_subgraph_size=min_subgraph_size_lookup[args.dataset],
        edge_imp=edge_imp,
        node_feat_imp=node_feat_imp,
        overall_node_feat_imp=overall_node_feat_imp,
        log_file=log_file,
        **explainer_eval_kwargs[args.explainer_type]
    )

    results_test = evaluate_explanation_partial(
        node_indices=node_index_test,
        log_prefix='test dataset all nodes: '
    )
    basis_results_test = evaluate_explanation_partial(
        node_indices=basis_node_index_test,
        log_prefix='test dataset basis nodes: '
    )
    motif_results_test = evaluate_explanation_partial(
        node_indices=motif_node_index_test,
        log_prefix='test dataset motif nodes: ',
    )

    io_utils.print_log(
        'Saving results to {}...'.format(output_file),
        log_file=log_file
    )
    with open(output_file, 'w') as txtfile:
        txtfile.write('test dataset all nodes\n')
        write_explainer_results(
            results_test,
            txtfile,
            prefix='\t'
        )
        txtfile.write('test dataset basis nodes\n')
        write_explainer_results(
            basis_results_test,
            txtfile,
            prefix='\t'
        )
        txtfile.write('test dataset motif nodes\n')
        write_explainer_results(
            motif_results_test,
            txtfile,
            prefix='\t'
        )
    io_utils.print_log('Done!', log_file=log_file)


if __name__ == '__main__':
    main()