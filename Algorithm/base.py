"""Shared plumbing for the algorithm implementations.

Each algorithm keeps its own logic under Algorithm/<Name>/; only the parts that
would otherwise be copy-pasted between them live here.
"""
import csv
import os


def apply_defaults(args, configs, name):
    """Fill in an algorithm's hyperparameters, honouring three levels of precedence:

        command line  >  the algorithm's config  >  the defaults in args.py

    `configs` holds a 'default' block that applies to every dataset, optionally
    layered with per-dataset overrides:

        CONFIGS = {
            'default': {'lr': 0.01, 'local_epochs': 1},
            'MNIST':   {'lr': 0.05},                     # only where a paper differs
        }

    A published hyperparameter has to override the generic default in args.py --
    otherwise a value the paper specifies never takes effect -- but must not
    override a value the user actually typed, or sweeps silently do nothing. Both
    failure modes existed before this helper: DESA/DFML/FedMD/FedProto set every
    key unconditionally (so `--lr 0.05` was discarded), while GeFL set keys only
    when absent (so any key args.py also defines -- weight_decay, momentum --
    could never be raised to the algorithm's own published value).

    The resolved values are printed, which also puts them in the run log; args.py
    values are logged by init_logging(), but these are settled after that runs.
    """
    config = dict(configs.get('default', {}))
    config.update(configs.get(args.dataset, {}))
    typed = getattr(args, '_cli_set', set())

    applied, kept = {}, {}
    for key, value in config.items():
        if key in typed:
            kept[key] = getattr(args, key)
        else:
            setattr(args, key, value)
            applied[key] = value

    print(f"[{name}] config: " + ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))
    if kept:
        print(f"[{name}] kept from command line: "
              + ", ".join(f"{k}={v}" for k, v in sorted(kept.items())))
    return args


def evaluate_all(clients, Client_Data, args, round_idx, recorder=None, model_list=None):
    """Evaluate every client and report per-client and average test metrics.

    Returns (avg_loss, avg_acc), matching the (loss, acc) order used everywhere else.

    Digit-5 is a domain-shift benchmark rather than a single test set, so it is
    routed to its own intra/inter-domain evaluation; the three algorithms that
    supported it each carried their own copy of this branch.
    """
    if getattr(args, 'dataset', '') == 'Digit5':
        from datasets.digit5.eval_digit5 import eval_digit5
        result = eval_digit5(clients, Client_Data['domain_test'],
                             Client_Data['client_domain'], args, round_idx=round_idx)
        if recorder is not None:
            recorder.record(round_idx, client_id=-1, model='all',
                            intra_acc=result['intra'], inter_acc=result['inter'],
                            worst_acc=result['worst'])
        return None, result['intra']

    total_loss, total_acc = 0.0, 0.0
    for client in clients:
        loss, acc = client.evaluate()
        total_loss += loss
        total_acc += acc
        cid = client.get_id()
        print(f'Round {round_idx}: Client {cid} - Test Acc: {acc:.4f}, Loss: {loss:.4f}')
        if recorder is not None:
            recorder.record(round_idx, client_id=cid,
                            model=model_list[cid] if model_list else '',
                            test_acc=acc, test_loss=loss)

    n = len(clients)
    avg_loss, avg_acc = total_loss / n, total_acc / n
    print(f'Round {round_idx}: Avg - Test Acc: {avg_acc:.4f}, Loss: {avg_loss:.4f}')
    if recorder is not None:
        recorder.record(round_idx, client_id=-1, model='avg',
                        test_acc=avg_acc, test_loss=avg_loss)
    return avg_loss, avg_acc


class MetricsRecorder:
    """Append per-round metrics to a CSV next to the run log.

    The log is human-readable but has to be re-parsed with a regex every time a
    figure is drawn. This writes the same numbers in a form that pandas can read
    directly, under the same directory and timestamp as the .log, so a run's text
    and its metrics stay paired.

    Rows are (round, client_id, ...), with client_id = -1 holding the average
    across clients for that round.
    """

    def __init__(self, args):
        prefix = getattr(args, 'run_prefix', None)
        if prefix is None:  # init_logging() was skipped, e.g. in a test harness
            self.path = None
            return
        self.path = prefix + '_metrics.csv'
        self._fields = None
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

    def record(self, round_idx, client_id, **metrics):
        if self.path is None:
            return
        row = {'round': round_idx, 'client_id': client_id, **metrics}
        write_header = self._fields is None
        if write_header:
            self._fields = list(row)
        with open(self.path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(row)
