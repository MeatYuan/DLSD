"""Default hyperparameters for DFML.

Reference: Khalil et al., "DFML: Decentralized Federated Mutual Learning", TMLR 2024.
Values follow the paper's implementation details (Appendix A.2): SGD with momentum
0.9 and weight decay 5e-4, mutual learning epochs K=10, temperature 1, and a cyclic
alpha whose period increments over time.
"""
from Algorithm.base import apply_defaults

DFML_CONFIGS = {
    'default': {
        'lr':                    0.01,
        'momentum':              0.9,
        'weight_decay':          5e-4,
        'local_epochs':          1,
        'dfml_K':                10,       # mutual learning epochs at the aggregator
        'dfml_T':                1.0,      # distillation temperature
        'dfml_alpha_min':        0.0,      # cyclic alpha lower bound
        'dfml_alpha_max':        0.8,      # upper bound; A.5.1 suggests 0.8-0.9 with WSM
        'dfml_init_period':      10,       # initial oscillation period
        'dfml_period_mode':      'linear',  # 'linear', 'mult', or 'fixed'
        'dfml_period_increment': 1,        # 'linear': period += this each cycle
        'dfml_period_mult':      2,        # 'mult': period *= this each cycle (SGDR style)
        'dfml_peak_updates':     3,        # None = derive from sample_rate (see _peak_updates)
        'dfml_kl_weighting':     'phi',    # 'phi' = Eq 3; 'uniform' = the A.4.7 variant
        'dfml_use_wsm':          True,     # WSM supervision signal (vs plain CE)
        'dfml_eval_peak':        True,     # report global accuracy from the peak models
    },
    # Per-dataset overrides go here; the paper uses the same values throughout.
}


def apply_dfml_defaults(args):
    """Apply the DFML hyperparameters (command line still wins)."""
    return apply_defaults(args, DFML_CONFIGS, name='DFML')
