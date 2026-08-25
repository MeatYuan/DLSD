"""Default hyperparameters for GeFL.

Reference: Kang et al., "Communicating with Grounding: Federated Learning with
Heterogeneous Data and Models", IEEE TMC 2025.
Values match the official repo: github.com/honggkang/hetero-model-fl-gen
"""
from Algorithm.base import apply_defaults

GEFL_CONFIGS = {
    'default': {
        'gefl_lr':       0.01,
        'gan_lr':        0.0002,
        'b1':            0.5,
        'b2':            0.999,
        'noise_dim':     100,
        'gan_epochs':    5,      # T_g=5 (local epochs for GM training)
        'local_ep_gen':  1,
        'local_ep':      5,
        'gen_wu_rounds': 100,    # T_KA=200 (100 warmup + 100 main), so warmup=100;
                                 # the official gen_wu_epochs is also 100 for MNIST
    },
    # Per-dataset overrides go here; the official repo uses the same values throughout.
}


def apply_gefl_defaults(args):
    """Apply the GeFL hyperparameters (command line still wins)."""
    return apply_defaults(args, GEFL_CONFIGS, name='GeFL')
