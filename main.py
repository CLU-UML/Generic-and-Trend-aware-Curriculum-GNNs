import os

import log
from train import train_model
from  evaluate import  eval_best_model

import argparse
import utils

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser('Interface for GTNN framework')

    parser.register('type', bool, utils.str2bool)  # add type keyword to registries

    parser.add_argument('--dataset', type=str, default='pgr', help='dataset name')

    # model hyperparameters
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 regularization weight')
    parser.add_argument('--nb_epoch', type=int, default=5, help="nb_epoch")
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

    # model configuration
    parser.add_argument('--fusion_type', type=str, default='outer', help='fusion_type')
    parser.add_argument('--training_type', type=str, default="regular", help="type of training as regular, curriculum (sl), curriculum with trend (sl_trend) ")
    parser.add_argument('--seed', type=int, default=123, help="seed")
    parser.add_argument('--model_type', type=str, default="sage", help="model type")

    # superloss hyperparameter
    parser.add_argument('--sl_lambda', type=float, default=10, help='superloss lambda')
    parser.add_argument('--prev_k_loss', type=int, default=10, help="previous k loss")
    parser.add_argument('--alpha', type=float, default=0.2, help="tune curriculum learning hyperparamenter")
    parser.add_argument('--mode', type=str, default='avg', help='mode')

    # dataset arguments
    parser.add_argument('--add_additional_feature', type=bool, default=True, help="to add or not to add additional feature")
    parser.add_argument('--embedding_type', type=str, default="doc2vec", help="node initilization embedding type")
    parser.add_argument('--neg_x', type=int, default=5, help='ratio of negative to positive examples')

    args = parser.parse_args()
    utils.fix_seed(args.seed)
    log.create_log(args)

    train_loader, val_loader, test_loader = utils.load_datasets(args)
    model = train_model(train_loader, val_loader,  args)
    eval_best_model(args, model, test_loader)


if __name__ == "__main__":
    main()
