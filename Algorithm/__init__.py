import importlib

ALGORITHMS = {
    'DLSD':     ('Algorithm.DLSD.main',     'DLSD'),
    'Local':    ('Algorithm.Local.main',    'Local'),
    'DESA':     ('Algorithm.DESA.main',     'DESA'),
    'DFML':     ('Algorithm.DFML.main',     'DFML'),
    'FedMD':    ('Algorithm.FedMD.main',    'FedMD'),
    'FedProto': ('Algorithm.FedProto.main', 'FedProto'),
    'GeFL':     ('Algorithm.GeFL.main',     'GeFL'),
}


def get_algorithm(name):
    """Return the entry point registered under `name`."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}. "
                         f"Available: {', '.join(sorted(ALGORITHMS))}")
    module_path, func_name = ALGORITHMS[name]
    return getattr(importlib.import_module(module_path), func_name)
