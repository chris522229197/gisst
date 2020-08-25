# Train, evaluate, and save a Graph Neural Network (GNN) model

import os
import sys
import argparse
from shutil import copyfile
import yaml
import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from sig.utils import io_utils
from sig.nn.models.gat import GAT
from sig.nn.models.gcn import GCN
from sig.nn.models.sigcn import SIGCN
from sig.utils.optimization_utils import train, batch_train, evaluate, batch_evaluate, \
    get_combo_sig_coeffs, tune_hyperparams


def parse_hyperparam(
    hyperparam_configs,
    delim,
    dtype
):
    """
    Parse the configs for a hyperparameter.

    Args:
        hyperparam_configs: Hyperparameter config object loaded from yaml.
        delim (str): Delimiter for the hyperparameter values.
        dtype (type): Data type for the hyperparameter values.
        
    Return:
        hyperparam_vals (list): Hyperparameter values.
    """
    hyperparam_vals = str(hyperparam_configs).split(delim)
    hyperparam_vals = [None if val == 'None' else dtype(val) for val in hyperparam_vals]
    return hyperparam_vals

def parse_configs(configs, **kwargs):
    """
    Parse the configuration dict loaded from a yaml file.
    
    Args:
        configs (dict): Configs loaded from a yaml file.
        
    Return:
        hyperparams (dict): Dict with hyperparameter names as keys and lists of 
            hyperparameter values as values.
    """
    dtypes = {
        'num_hidden_layers': int,
        'hidden_dims': int,
        'learning_rates': float,
        'nums_epochs': int,
        'l2_coeffs': float,
        'dropout_rates': float,
        'x_l1_coeffs': float,
        'x_ent_coeffs': float,
        'edge_l1_coeffs': float,
        'edge_ent_coeffs': float
    }
    
    hyperparams = configs['hyperparams']
    for name, dtype in dtypes.items():
        hyperparams[name] = parse_hyperparam(
            hyperparam_configs=hyperparams[name],
            dtype=dtype,
            **kwargs
        )
    return hyperparams

def write_hyperparams(
    hyperparams,
    outfile,
    prefix=''
):
    """
    Write model hyperparameters to a text file.

    Args:
        hyperparams (dict): Hyperparameter names and values.
        outfile (file object): Output file.
        prefix (str): Prefix for each metric name.

    Return:
        No objects returned.
    """
    for name, value in hyperparams.items():
        outfile.write(
            '{}{}: {}\n'.format(
                prefix,
                name,
                value
            ) 
        )

def write_results(
    results,
    outfile, 
    prefix=''
):
    """
    Write model evaluation results to a text file.

    Args:
        results (dict): Performance metric names and values.
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
                'None' if value is None else '%0.4f' % value
            )
        )

DATASET_LOADER_USAGE = { # whether the dataset requires data loader
    'ba_300_house_80': False,
    'ba_300_house_80_comm': False,
    'tree_8_cycle_80': False,
    'tree_8_grid_80': False,
    'ba_300_cycle_80': False,
    'ba_300_grid_80': False,
    'tree_8_house_80': False,
    'mutagenicity': True,
    'reddit_binary': True,
    'proteins': True
}

DATASET_FILENAME = { # filename (no extension) for each dataset
    'ba_300_house_80': 'data',
    'ba_300_house_80_comm': 'data',
    'tree_8_cycle_80': 'data',
    'tree_8_grid_80': 'data',
    'ba_300_cycle_80': 'data',
    'ba_300_grid_80': 'data',
    'tree_8_house_80': 'data',
    'mutagenicity': 'dataset',
    'reddit_binary': 'dataset',
    'proteins': 'dataset'
}

DATASET_CLASSIFY_GRAPH = { # whether the dataset is for graph classification
    'ba_300_house_80': False,
    'ba_300_house_80_comm': False,
    'tree_8_cycle_80': False,
    'tree_8_grid_80': False,
    'ba_300_cycle_80': False,
    'ba_300_grid_80': False,
    'tree_8_house_80': False,
    'mutagenicity': True,
    'reddit_binary': True,
    'proteins': True
}

def main():
    parser = argparse.ArgumentParser(
        description="train, evaluate, and save a GNN model"
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
            'tree_8_house_80',
            'mutagenicity',
            'reddit_binary',
            'proteins'
        ],
        help='dataset to use'
    )
    parser.add_argument(
        'model_type',
        type=str,
        choices=[
            'gcn',
            'gat',
            'sigcn'
        ],
        help='type of GNN model to use'
    )
    parser.add_argument(
        'configs_yaml',
        type=str,
        help='config yaml filename'
    )
    parser.add_argument(
        'criterion',
        type=str,
        choices=['auroc', 'accuracy'],
        help='criterion for hyperparameter tuning'
    )
    parser.add_argument(
        'outdir',
        type=str,
        help='output directory with trailing /'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='batch size for batch optimization using dataloader',
        dest='batch_size'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='random seed',
        dest='seed'
    )

    args = parser.parse_args()
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

    sub_outdir = args.outdir + args.dataset + '/'
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)
    subsub_outdir = sub_outdir + args.model_type + '/'
    if not os.path.exists(subsub_outdir):
        os.makedirs(subsub_outdir)
    existing_index = [
        int(filedir.replace('model_output_', '')) \
            for filedir in os.listdir(subsub_outdir) \
            if filedir.startswith('model_output_')
    ]
    if len(existing_index) == 0:
        final_index = 0
    else:
        final_index = max(existing_index) + 1
    final_outdir = subsub_outdir + 'model_output_{}'.format(final_index) + '/'
    os.makedirs(final_outdir)
    copyfile(args.configs_yaml, final_outdir + 'configs.yaml')
    
    log_file = final_outdir + 'log.txt'
    io_utils.print_log('seed={}'.format(args.seed), log_file=log_file, mode='w')
    io_utils.print_log('device={}'.format(device.type), log_file=log_file)
    io_utils.print_log('Loading in data...', log_file=log_file)
    data_file = 'graphs/{}/{}.pyg.pkl'.format(
        args.dataset, 
        DATASET_FILENAME[args.dataset]
    )
    with open(args.configs_yaml, 'r') as yamlfile:
        configs = yaml.safe_load(yamlfile)
    hyperparams = parse_configs(configs, delim=',')
    num_hidden_layers = hyperparams['num_hidden_layers']
    hidden_dims = hyperparams['hidden_dims']
    learning_rates = hyperparams['learning_rates']
    nums_epochs = hyperparams['nums_epochs']
    l2_coeffs = hyperparams['l2_coeffs']
    dropout_rates = hyperparams['dropout_rates']
    x_l1_coeffs = hyperparams['x_l1_coeffs']
    x_ent_coeffs = hyperparams['x_ent_coeffs']
    edge_l1_coeffs = hyperparams['edge_l1_coeffs']
    edge_ent_coeffs = hyperparams['edge_ent_coeffs']

    data = torch.load(data_file) # actually a dataset for using dataloader
    classify_graph = DATASET_CLASSIFY_GRAPH[args.dataset]

    if DATASET_LOADER_USAGE[args.dataset]:
        fn_train = batch_train
        fn_eval = batch_evaluate
        kwargs_train = {
            'loader': DataLoader(
                data[data.index_train],
                args.batch_size,
                shuffle=True, 
                num_workers=0
            ),
            'device': device
        }
        kwargs_val = {
            'loader': DataLoader(
                data[data.index_val],
                args.batch_size,
                shuffle=True,
                num_workers=0
            ),
            'device': device
        }
        kwargs_test = {
            'loader': DataLoader(
                data[data.index_test],
                args.batch_size,
                shuffle=True,
                num_workers=0
            ),
            'device': device
        }
        for data_train in kwargs_train['loader']:
            input_size = data_train.x.shape[1]
            break
        num_classes = len(set(data.data.y.tolist()))
    else:
        data = data.to(device)
        fn_train = train
        fn_eval = evaluate
        kwargs_train = {
            'x': data.x,
            'edge_index': data.edge_index[:, data.edge_mask_train],
            'y': data.y_train,
            'mask': data.node_mask_train
        }
        kwargs_val = {
            'x': data.x,
            'edge_index': data.edge_index[:, data.edge_mask_val],
            'y': data.y_val,
            'mask': data.node_mask_val
        }
        kwargs_test = {
            'x': data.x,
            'edge_index': data.edge_index[:, data.edge_mask_test],
            'y': data.y_test,
            'mask': data.node_mask_test
        }
        input_size = data.x.shape[1]
        num_classes = len(set(data.y_train.tolist()))

    io_utils.print_log('Tuning the hyperparameters...', log_file=log_file)
    best_model, \
    best_hyperparams, \
    best_results_train, \
    best_results_val = tune_hyperparams(
        model_type=args.model_type.upper(), 
        input_size=input_size,
        output_size=num_classes,
        num_hidden_layers=num_hidden_layers,
        hidden_dims=hidden_dims, 
        learning_rates=learning_rates,
        nums_epochs=nums_epochs,
        l2_coeffs=l2_coeffs,
        dropout_rates=dropout_rates,
        device=device,
        x_l1_coeffs=x_l1_coeffs,
        x_ent_coeffs=x_ent_coeffs,
        edge_l1_coeffs=edge_l1_coeffs,
        edge_ent_coeffs=edge_ent_coeffs,
        criterion=args.criterion,
        kwargs_train=kwargs_train,
        kwargs_eval=kwargs_val,
        fn_train=fn_train,
        fn_eval=fn_eval,
        verbosity=2,
        name_train='train',
        name_eval='val', 
        log_file=log_file,
        classify_graph=classify_graph
    )
    best_sig_coeffs = get_combo_sig_coeffs(best_hyperparams)

    io_utils.print_log('Evaluating on the test set...', log_file=log_file)
    best_results_test = fn_eval(
        model=best_model,
        sig_coeffs=best_sig_coeffs,
        **kwargs_test
    )

    io_utils.print_log('Saving the output...', log_file=log_file)
    best_state_dict_file = final_outdir + 'best_state_dict.pt'
    best_results_file = final_outdir + 'best_results.txt'
    best_hyperparams_file = final_outdir + 'best_hyperparams.txt'

    torch.save(best_model.state_dict(), best_state_dict_file)
    with open(best_results_file, 'w') as txtfile:
        txtfile.write('train\n')
        write_results(
            best_results_train, 
            txtfile,
            '\t'
        )
        txtfile.write('val\n')
        write_results(
            best_results_val, 
            txtfile,
            '\t'
        )
        txtfile.write('test\n')
        write_results(
            best_results_test, 
            txtfile,
            '\t'
        )
    with open(best_hyperparams_file, 'w') as txtfile:
        write_hyperparams(best_hyperparams, txtfile)
    io_utils.print_log('Done!', log_file=log_file)


if __name__ == '__main__':
    main()