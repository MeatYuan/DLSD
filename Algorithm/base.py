"""Shared plumbing for the algorithm implementations.

Each algorithm keeps its own logic under Algorithm/<Name>/; only the parts that
would otherwise be copy-pasted between them live here.
"""
import csv
import os

from utils.train import Evaluate


class Losses:
    """Weighted running means, keyed by name.

    A training loop that reports k losses otherwise needs four blocks of
    bookkeeping -- declare the epoch totals, accumulate them, divide each by its
    sample count, average those over the epochs -- and adding one more loss means
    touching all four. Here `add` carries the sample count with the value, so each
    metric keeps its own denominator and the epoch layer disappears entirely:
    accumulating across every epoch and dividing once at the end gives the same
    number as averaging the per-epoch means, as long as the epochs see the same
    number of samples (they do -- it is the same DataLoader).

    Keeping the count per metric also removes a silent failure mode. Metrics
    measured over different batches -- the loss on the proxy data and the loss on
    the local data -- need different denominators, and pairing them by hand is
    something only a careful reader would catch.

        losses = Losses('loss', 'acc', 'kd')
        losses.add('loss', loss.item(), inputs.size(0))   # value is a per-sample mean
        losses['loss']                                    # -> weighted mean

    The names are declared up front and an unknown one raises: a mistyped key would
    otherwise open a new metric that silently reads back as 0.0.
    """

    def __init__(self, *names):
        self._names = list(names)
        self._sum = {name: 0.0 for name in self._names}
        self._count = {name: 0.0 for name in self._names}

    def _check(self, name):
        if name not in self._sum:
            raise KeyError(f"Unknown metric {name!r}. Declared: {', '.join(self._names)}")

    def add(self, name, value, n=1):
        """Record `value`, the mean over `n` samples."""
        self._check(name)
        self._sum[name] += value * n
        self._count[name] += n

    def merge(self, other, n=1):
        """Fold another Losses in as a single observation of weight `n`.

        Used to average client-level results into a round-level summary: every
        client counts once, regardless of how many samples it trained on.
        """
        for name in other._names:
            self._check(name)
            self.add(name, other[name], n)

    def __getitem__(self, name):
        self._check(name)
        count = self._count[name]
        return self._sum[name] / count if count else 0.0


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


class BaseClient:
    """State and reporting shared by every algorithm's client.

    Subclasses own the training; this holds what all of them keep anyway, plus the
    two methods that were otherwise copied verbatim into each one. Anything an
    algorithm needs on top -- a peak model, an adapter, a prototype cache -- belongs
    in its own __init__ after the super() call.
    """

    def __init__(self, model, local_train_loader, local_test_loader, client_id,
                 args, device='cpu'):
        self.client_id = client_id
        self.local_model = model
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader
        self.args = args
        self.device = device

    def get_id(self):
        return self.client_id

    def print_info(self, **extra):
        """One line per client. The seven-line box this replaces printed 210 lines
           before training even started, on a 30-client run."""
        fields = {'model': self.local_model.__class__.__name__,
                  'train': len(self.train_loader.dataset),
                  'test': len(self.test_loader.dataset),
                  'device': self.device,
                  **extra}
        print(f"Client {self.client_id:>3} | "
              + " | ".join(f"{k}={v}" for k, v in fields.items()))

    def evaluate(self):
        """Loss and accuracy on the shared test set."""
        return Evaluate(self.local_model, self.test_loader, device=self.device)
