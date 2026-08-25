"""Default hyperparameters for FedMD.

Reference: D. Li and J. Wang, "FedMD: Heterogenous Federated Learning via Model
Distillation", NeurIPS 2019 workshop.
Modified with reference to: https://github.com/diogenes0319/FedMD_clean
"""
from Algorithm.base import apply_defaults

FEDMD_CONFIGS = {
    'default': {
        'lr':                     1e-2,
        'public_pretrain_epochs': 25,
        'pretrain_epochs':        0,
        'digest_epochs':          1,
        'revisit_epochs':         5,
        'public_size':            5000,
        'fedmd_T':                20.0,
    },
    # Per-dataset overrides go here; the paper uses the same values throughout.
}


def apply_fedmd_defaults(args):
    """Apply the FedMD hyperparameters (command line still wins)."""
    return apply_defaults(args, FEDMD_CONFIGS, name='FedMD')
