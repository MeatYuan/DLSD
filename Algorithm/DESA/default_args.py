"""Default hyperparameters for DESA.

Reference implementation: https://github.com/ubc-tea/DESA
(the synthetic-data routines in utils/syn.py are taken from iterative_desa.py).
"""
from Algorithm.base import apply_defaults

DESA_CONFIGS = {
    'default': {
        'lr':               0.01,
        'local_epochs':     1,
        'IPC':              50,
        'syn_batch_size':   256,
        'syn_iters':        1000,
        'lambda_ori':       1.0,
        'lambda_kd':        1.0,
        'lambda_reg':       1.0,
    },
    # Per-dataset overrides go here; the paper uses the same values throughout.
}


def apply_desa_defaults(args):
    """Apply the DESA hyperparameters (command line still wins)."""
    return apply_defaults(args, DESA_CONFIGS, name='DESA')
