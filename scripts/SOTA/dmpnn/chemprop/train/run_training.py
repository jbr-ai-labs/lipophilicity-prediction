from logging import Logger
import os
import sys
from typing import List
import pandas as pd


import numpy as np
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from args import TrainArgs
from constants import MODEL_FILE_NAME
from data import get_class_sizes, get_data, MoleculeDataLoader, split_data, StandardScaler, validate_dataset_type
from models import MoleculeModel
from nn_utils import param_count
from utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,scaler, features_scaler, args, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_rmses = None
        self.best_r2s = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.epoch = 0

            
        self.scaler = scaler
        self.features_scaler = features_scaler
        self.args = args
    def __call__(self, val_loss, val_rmses, val_r2s, model, epoch = 0):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_rmses = val_rmses
            self.best_r2s = val_r2s
            self.save_chckpt(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.epoch = epoch
            print ('EarlyStopping counter: ',self.counter,  ' out of ', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_rmses = val_rmses
            self.best_r2s = val_r2s
            self.save_chckpt(val_loss, model)
            self.counter = 0

    def save_chckpt(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print ('Validation loss decreased', round(self.val_loss_min,6),' -->', round(val_loss,6), '.  Saving model ...')
        save_checkpoint(self.path, model, self.scaler, self.features_scaler, self.args)
        self.val_loss_min = val_loss


def run_training(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param logger: A logger to record output.
    :return: A list of model scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    
    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    args.save(os.path.join(args.save_dir, 'args.json'))

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.0, 0.2), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            data_path=args.data_path,
            save_dir=args.save_dir,
            smiles_column=args.smiles_column
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        cache = True
        num_workers = 0
    else:
        cache = False
        num_workers = args.num_workers
    from torch.utils.data import  Dataset, DataLoader
    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        #         best_score = float('inf') if args.minimize_score else -float('inf')
        #         best_score_R2 = 0
        #         best_epoch, n_iter = 0, 0

        n_iter = 0

        early_stopping = EarlyStopping(scaler, features_scaler, args, patience=args.patience, delta = args.delta, \
                                       verbose=True, path = os.path.join(save_dir, MODEL_FILE_NAME))
        
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, val_scores_r2 = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger,
                args = args
            )

            # Average validation score
            avg_val_score = val_scores
            avg_val_score_r2 = val_scores_r2
            print(avg_val_score)
            for task in range(args.num_tasks):
                debug(f'Validation {args.metric} {args.target_columns[task]} = {avg_val_score[task]:.6f}')
                debug(f'Validation R2 {args.target_columns[task]} = {avg_val_score_r2[task]:.6f}')

                writer.add_scalar(f'validation_{args.metric}_{args.target_columns[task]}', avg_val_score[task], n_iter)
                writer.add_scalar(f'validation_R2_{args.target_columns[task]}', avg_val_score_r2[task], n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)
                    
            avg_val_score = np.nanmean(val_scores)
            avg_val_score_r2 = np.nanmean(val_scores_r2)
            
            early_stopping(avg_val_score, val_scores, val_scores_r2, model, epoch)

            # Save model checkpoint if improved validation score
#             if args.minimize_score and avg_val_score < best_score or \
#                     not args.minimize_score and avg_val_score > best_score:
#                 best_score, best_epoch = avg_val_score, epoch
#                 best_score_r2 = avg_val_score_r2
#                 best_vals = val_scores
#                 best_vals_r2 = val_scores_r2
#                 save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler, args)
            
            if early_stopping.early_stop:
                print("Early stopping", epoch)
                break
            
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {early_stopping.best_score:.6f} on epoch {early_stopping.best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler,
            args = args
        )
        test_predictions = pd.DataFrame(columns = [args.smiles_column])
        test_predictions[args.smiles_column] = test_smiles
        for task in range(args.num_tasks):
            test_predictions[args.target_columns[task]] = np.array(test_targets)[:, task]
            test_predictions[args.target_columns[task]+'_pred'] = np.array(test_preds)[:, task]
        test_predictions.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index = False)
        
        test_scores, test_scores_r2 = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        
        avg_test_score = test_scores
        avg_test_score_r2 = test_scores_r2
#         print(avg_test_score)
        for task in range(args.num_tasks):
            info(f'Model {model_idx} test {args.metric} {args.target_columns[task]} = {avg_test_score[task]:.6f}')
            info(f'Model {model_idx} test R2 {args.target_columns[task]} = {avg_test_score_r2[task]:.6f}')
            writer.add_scalar(f'test_{args.metric}_{args.target_columns[task]}', avg_test_score[task], 0)
            writer.add_scalar(f'test_R2_{args.target_columns[task]}', avg_test_score_r2[task], 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)
        writer.close()

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores, ensemble_scores_r2 = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    
    avg_ensemble_test_score = ensemble_scores
    avg_ensemble_test_score_r2 = ensemble_scores_r2

    for task in range(args.num_tasks):
        info(f'Ensemble test {args.metric}  {args.target_columns[task]}= {avg_ensemble_test_score[task]:.6f}')
        info(f'Ensemble test R2  {args.target_columns[task]}= {avg_ensemble_test_score_r2[task]:.6f}')

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores, ensemble_scores_r2, early_stopping.best_rmses, early_stopping.best_r2s
