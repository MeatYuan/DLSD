import argparse
import sys
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='DLSD', help='Algorithm to use (e.g., DLSD, DFML, DESA)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use (MNIST or CIFAR10)')
    parser.add_argument('--Model_Heterogeneity', action=argparse.BooleanOptionalAction, default = True, help='Model heterogeneity or not')
    parser.add_argument('--num_clients', type=int, default=30, help='Number of clients')
    parser.add_argument('--noniid_alpha', type=float, default=10, help='Dirichlet alpha for non-IID data split')
    # Rounds and epochs
    parser.add_argument('--Round', type=int, default=200, help='Number of device-to-device communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs per client')
    
    # Learning parameters 
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use (SGD or Adam)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    #adapter learning parameters
    parser.add_argument('--with_adapter', action=argparse.BooleanOptionalAction, default = True, help='Train with adapter or not')
    parser.add_argument('--Adapter_agg', action=argparse.BooleanOptionalAction, default = False, help='Aggregate Adapter')
    parser.add_argument('--Evaluate_with_adapter', action=argparse.BooleanOptionalAction, default = False, help='Evaluate with adapter or not')
    parser.add_argument('--adapter_type', type=str, default='mlp', help='Type of adapter layer (conv or mlp)')
    parser.add_argument('--global_kl_weight', type=float, default=1.0, help='Distillation loss weight for the client network')
    parser.add_argument('--global_mse_weight', type=float, default=1.0, help='MSE loss weight for the client network')
    parser.add_argument('--global_local_weight', type=float, default=1.0, help='Local loss weight for the client network')
    parser.add_argument('--Distallation_T', type=float, default=5.0, help='Temperature for distillation loss')
    parser.add_argument('--train_adapter_mse_weight', type=float, default=1.0, help='MSE loss weight for adapter training')
    parser.add_argument('--train_adapter_decoder_weight', type=float, default=1.0, help='adapter decoder loss weight for adapter training')
    # Client-network parameters
    parser.add_argument('--sample_rate', type=float, default=0.4, help='Fraction of clients participating each round')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    # Seed
    parser.add_argument('--Noniid_random_state_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--random_seed', type=int, default=5, help='Random seed for reproducibility')
    
    #TEST args
    parser.add_argument('--IPC', type=int, default=50, help='Images Per Class for synthetic data generation')
    parser.add_argument('--syn_lr', type=float, default=1.0, help='Learning rate for synthetic data generation')
    parser.add_argument('--syn_momentum', type=float, default=0.5, help='Momentum for synthetic data generation')
    parser.add_argument('--syn_iters', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--syn_batch_size', type=int, default=128, help='Batch size for synthetic data generation')
    parser.add_argument('--syn_round', type=int, default=30, help='Number of rounds for synthetic data generation')
    parser.add_argument('--samples_per_domain', type=int, default=7000, help='Digit5: cap each domain to this many samples (equalise domain sizes so data quantity is not a confound)')

    args = parser.parse_args()

    # Record which options were actually typed on the command line, so that an
    # algorithm's own defaults (Algorithm/<Name>/default_args.py) can fill in the
    # rest without ever overriding an explicit choice. Reading sys.argv rather than
    # comparing against the parser defaults keeps this correct even when the value
    # the user passed happens to equal the default.
    typed = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv or any(a.startswith(opt + '=') for a in sys.argv):
                typed.add(action.dest)
    args._cli_set = typed

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.image_size = (32, 32)
        args.image_channel = 3

    elif args.dataset == 'MNIST':
        args.num_classes = 10
        args.image_size = (32, 32)
        args.image_channel = 1
    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.image_size = (32, 32)
        args.image_channel = 3
    elif args.dataset == 'Digit5':
        args.num_classes = 10
        args.image_size = (32, 32)
        args.image_channel = 3

    if args.with_adapter == False:
        args.Evaluate_with_adapter = False
        args.Adapter_agg = False

    if args.Adapter_agg == True:
        if args.adapter_type == "conv":
            args.Adapter_agg = False
            print('Adapter conv cant aggregate.')

    return args 