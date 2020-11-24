"""Optimizes hyperparameters using Bayesian optimization."""

import inspect
import os
import sys
from functools import reduce

import copy

import yaml

from chemprop.train import cross_validate

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from .args import TrainArgs

SPACE = {
    'hidden_size': [300, 600, 900],
    'substructure_hidden_size': [300, 600, 900],
    'depth': [4, 6],
    'ffn_num_layers': [1, 3]
}


def grid_search_hyperopt(args: TrainArgs):
    start_save_dir = args.save_dir
    scores = {}
    for hs in SPACE['hidden_size']:
        for shs in SPACE['substructure_hidden_size']:
            for depth in SPACE['depth']:
                for ffn_num_layers in SPACE['ffn_num_layers']:
                    current_args = copy.copy(args)
                    if not current_args.additional_encoder:
                        shs = 0
                    cur_args = {
                        'hidden_size': str(hs),
                        'substructure_hidden_size': str(shs),
                        'depth': str(depth),
                        'ffn_num_layers': str(ffn_num_layers)
                    }
                    params_suffix = '_'.join(reduce(lambda x, y: x + y, cur_args.items()))
                    if params_suffix in scores.keys():
                        continue
                    print(current_args.data_path)
                    current_args.hidden_size = hs
                    current_args.substructures_hidden_size = shs
                    current_args.depth = depth
                    current_args.ffn_num_layers = ffn_num_layers
                    current_args.save_dir = start_save_dir + '/' + params_suffix
                    mean_scores, std_scores = cross_validate(args=current_args)
                    scores[params_suffix] = [mean_scores.tolist(), std_scores.tolist()]
                    with open(start_save_dir + '/results.yaml', 'w') as file:
                        yaml.dump(scores, file)


def chemprop_grid_search() -> None:
    grid_search_hyperopt(args=TrainArgs().parse_args())
