import torch
import random
import numpy as np
import logging
import sys
from datetime import datetime
import os

def set_seed(args):
    random.seed(args.random_seed)  # Use args.random_state_seed instead of a fixed seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logging(args,):
    log_prefix = getattr(args, 'log_dir', "")
    verbose = getattr(args, 'verbose', True)

    folder = f'result/{args.algorithm}/{args.dataset}/{args.noniid_alpha}/'
    folder = log_prefix + folder

    file_prefix = f'{folder}/{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    log_filename = file_prefix + ".log"

    os.makedirs(folder, exist_ok=True)

    # Published so that anything else writing per-run output (the metrics CSV,
    # saved models, sample images) lands beside the log under the same timestamp,
    # instead of computing its own and drifting a second or two apart.
    args.run_prefix = file_prefix

    handlers=[logging.FileHandler(log_filename)]

    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers= handlers
    )

    class PrintLogger:
        def write(self, message):
            if message.strip():
                logging.info(message.strip())
        def flush(self):
            pass

    sys.stdout = PrintLogger()
    if verbose:
        sys.stderr = PrintLogger()

    logging.info(f"Logging initialized, outputting to {log_filename}")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")