"""Default hyperparameters for FedProto.

Reference: Tan et al., "FedProto: Federated Prototype Learning across
Heterogeneous Clients", AAAI 2022. https://github.com/yuetan031/FedProto
"""
from Algorithm.base import apply_defaults

FEDPROTO_CONFIGS = {
    'default': {
        'ld':               10.0,
        'lr':               0.01,
        'momentum':         0.5,
        'train_ep':         1,
        'proto_dim':        256,
    },
    # Per-dataset overrides go here; the paper uses the same values throughout.
}


def apply_fedproto_defaults(args):
    """Apply the FedProto hyperparameters (command line still wins)."""
    return apply_defaults(args, FEDPROTO_CONFIGS, name='FedProto')
