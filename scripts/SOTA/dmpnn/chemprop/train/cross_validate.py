import csv
import os
from typing import Tuple

import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))
sys.path.insert(0,parentdir) 

from .run_training import run_training
from args import TrainArgs
from constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from data import get_task_names
from utils import create_logger, makedirs, timeit

import json


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation for a Chemprop model.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    info = logger.info if logger is not None else print
    
    if args.separate_test_path!='' and args.separate_val_path=='':
        DATASET_PATH = args.separate_test_path
        DATASET_OUTPUT_PATH = os.path.join(project_path,'./data/raw/baselines/dmpnn')
        dataset_train = pd.read_csv(os.path.join(DATASET_PATH, args.file_prefix+'_train.csv'), index_col=0)
        dataset_val = pd.read_csv(os.path.join(DATASET_PATH, args.file_prefix+'_validation.csv'), index_col=0)
        dataset_train_val = pd.concat([dataset_train, dataset_val], axis = 0).reset_index(drop = True)
        dataset_train_val.to_csv(os.path.join(DATASET_OUTPUT_PATH,  args.file_prefix+'_train_val_dataset.csv'),index = False)
    
    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_column=args.smiles_column,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns
    )
    # Run training on different random seeds for each fold
    all_scores = []
    all_scores_r2 = []
    val_scores = []
    val_scores_r2 = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores, model_scores_r2, val_score, val_score_r2 = run_training(args, logger)
        all_scores.append(model_scores)
        all_scores_r2.append(model_scores_r2)
        val_scores.append(val_score)
        val_scores_r2.append(val_score_r2)
    all_scores = np.array(all_scores)
    all_scores_r2 = np.array(all_scores_r2)
    val_scores = np.array(val_scores)
    val_scores_r2 = np.array(val_scores_r2)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'\tSeed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')
        info(f'\tSeed {init_seed + fold_num} ==> test R2 = {np.nanmean(all_scores_r2[fold_num]):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} R2 = {all_scores_r2[fold_num]:.6f}')

    # Report scores across models
#     avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(all_scores, axis = 0), np.nanstd(all_scores, axis=0)
    
#     avg_val_scores = np.nanmean(val_scores)  # average score for each model across tasks
#     avg_val_scores = np.nanmean(val_scores, axis=1)  # average score for each model across tasks
    mean_val_score, std_val_score = np.nanmean(val_scores, axis = 0), np.nanstd(val_scores, axis = 0)
    
#     avg_scores_r2 = np.nanmean(all_scores_r2, axis=1)  # average score for each model across tasks
    mean_score_r2, std_score_r2 = np.nanmean(all_scores_r2, axis = 0), np.nanstd(all_scores_r2, axis = 0)
    
#     avg_val_scores_r2 = np.nanmean(val_scores_r2, axis=1)  # average score for each model across tasks
    mean_val_score_r2, std_val_score_r2 = np.nanmean(val_scores_r2, axis = 0), np.nanstd(val_scores_r2, axis = 0)
    
    for task in range(args.num_tasks):
        info(f'Overall val {args.metric} {args.target_columns[task]}= {mean_val_score[task]:.6f} +/- {std_val_score[task]:.6f}')
        info(f'Overall val R2 {args.target_columns[task]} = {mean_val_score_r2[task]:.6f} +/- {std_val_score_r2[task]:.6f}')
        info(f'Overall test {args.metric} {args.target_columns[task]} = {mean_score[task]:.6f} +/- {std_score[task]:.6f}')
        info(f'Overall test R2 {args.target_columns[task]} = {mean_score_r2[task]:.6f} +/- {std_score_r2[task]:.6f}')
    
    all_scores_dict = {}
    for task in range(args.num_tasks):
        all_scores_dict[args.target_columns[task]+'_test_RMSE_mean'] = mean_score[task]
        all_scores_dict[args.target_columns[task]+'_test_R2_mean'] = mean_score_r2[task]
        all_scores_dict[args.target_columns[task]+'_test_RMSE_std'] = std_score[task]
        all_scores_dict[args.target_columns[task]+'_test_R2_std'] = std_score_r2[task]
        all_scores_dict[args.target_columns[task]+'_val_RMSE_mean'] = mean_val_score[task]
        all_scores_dict[args.target_columns[task]+'_val_R2_mean'] = mean_val_score_r2[task]
        all_scores_dict[args.target_columns[task]+'_val_RMSE_std'] = std_val_score[task]
        all_scores_dict[args.target_columns[task]+'_val_R2_std'] = std_val_score_r2[task]
    
    with open(os.path.join(os.path.dirname(save_dir), 'final_scores.json'), 'w') as f:
        json.dump(all_scores_dict, f)
    
    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'\tOverall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', f'Mean {args.metric}', f'Standard deviation {args.metric}'] +
                        [f'Fold {i} {args.metric}' for i in range(args.num_folds)])

        for task_num, task_name in enumerate(args.task_names):
            task_scores = all_scores[:, task_num]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            writer.writerow([task_name, mean, std] + task_scores.tolist())

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    cross_validate(args=TrainArgs().parse_args())
