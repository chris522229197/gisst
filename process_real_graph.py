# Process a real-world graph dataset for experiments

import os
import sys
import argparse
import torch
from sig.utils import io_utils
from sig.utils import graph_utils

def main():
    parser = argparse.ArgumentParser(
        description="process a real-world graph dataset"
    )

    parser.add_argument(
        'outdir',
        type=str,
        help='output directory with trailing /'
    )
    parser.add_argument(
        'name', 
        type=str, 
        choices=[
            'mutagenicity', 
            'reddit_binary',
            'proteins'
        ],
        help='name for the real-world graph dataset'
    )
    parser.add_argument(
        '--fold-seed',
        type=int,
        default=456,
        help='random seed for splitting the graph into train, val, and test folds',
        dest='fold_seed'
    )

    tu_name_lookup = {
        'mutagenicity': 'Mutagenicity',
        'reddit_binary': 'REDDIT-BINARY',
        'proteins': 'PROTEINS'
    }

    args = parser.parse_args()
    print('Running {} with arguments'.format(sys.argv[0]))
    for arg in vars(args):
        print('\t{}={}'.format(arg, getattr(args, arg)))

    dataset = graph_utils.get_dataset(tu_name_lookup[args.name])
    y = dataset.data.y.tolist()
    index = [i for i in range(len(y))]
    index_train, index_val, index_test, \
    y_train, y_val, y_test = graph_utils.split_index_train_val_test(
        index,
        y,
        seed=args.fold_seed
    )

    dataset.index_train = torch.LongTensor(index_train)
    dataset.index_val = torch.LongTensor(index_val)
    dataset.index_test = torch.LongTensor(index_test)

    dist_train = io_utils.get_result_mesg(
        graph_utils.calc_dataset_dist(dataset[dataset.index_train])
    )
    dist_val = io_utils.get_result_mesg(
        graph_utils.calc_dataset_dist(dataset[dataset.index_val])
    )
    dist_test = io_utils.get_result_mesg(
        graph_utils.calc_dataset_dist(dataset[dataset.index_test])
    )
    dists = {
        'train': dist_train,
        'val': dist_val,
        'test': dist_test
    }

    sub_outdir = args.outdir + args.name + '/'
    if not os.path.exists(sub_outdir):
        os.makedirs(sub_outdir)

    print('Saving the output to {}...'.format(sub_outdir))
    torch.save(dataset, sub_outdir + 'dataset.pyg.pkl')

    print('Saving the log file to {}...'.format(sub_outdir))
    with open(sub_outdir + 'log.txt', 'w') as f:
        for split, dist in dists.items():
            f.write('{}\n'.format(split))
            for value in dist.values():
                f.write('\t{}\n'.format(value))
    print('Finished running {}'.format(sys.argv[0]))


if __name__ == '__main__':
    main()