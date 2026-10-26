# Decentralised Federated Learning with Heterogeneous Models

Every client owns a different architecture, so parameter averaging is unavailable and
knowledge has to move between clients some other way. This repository holds the
proposed method and the baselines it is compared against, all run through one entry
point on an identical data split, model assignment, and client-participation sequence.

## Environment

All results were produced with the following versions:

| Component | Version |
| --- | --- |
| Python | 3.12.7 |
| PyTorch | 2.10.0 |
| CUDA | 13.0 |

## Running

```bash
python main.py --algorithm DLSD --dataset CIFAR10 --noniid_alpha 0.1 --device cuda:0
```

`--algorithm` accepts any key in the registry (see below). `python main.py --help`
lists every option; the ones that matter most:

| Option | Meaning |
| --- | --- |
| `--dataset` | `CIFAR10`, `MNIST`, `SVHN`, or `Digit5` |
| `--noniid_alpha` | Dirichlet concentration for the label split. `0.1` is a heavy label shift, `10` is close to IID |
| `--num_clients` / `--Round` | Number of clients and communication rounds |
| `--sample_rate` | Fraction of clients participating each round |
| `--Model_Heterogeneity` / `--no-Model_Heterogeneity` | Mixed architectures, or one architecture for everyone |

Datasets download themselves into `Data/` on first use.

## Algorithms

| Key | Reference |
| --- | --- |
| `DLSD` | This work: distribution-matched synthetic images plus an adapter shared over the client network |
| `Local` | No communication; the lower bound |
| `DESA` | C.-Y. Huang, K. Srinivas, X. Zhang, and X. Li, “Overcoming data and
model heterogeneities in decentralized federated learning via synthetic
anchors,” ICML, 2024. |
| `DFML` | Y. H. Khalil, A. H. Estiri, M. Beitollahi, N. Asadi, S. Hemati,
X. Li, G. Zhang, and X. Chen, “DFML: Decentralized federated
mutual learning,” Transactions on Machine Learning Research, 2024. |
| `FedMD` | D. Li and J. Wang, “Fedmd: Heterogenous federated learning via model
distillation,” arXiv preprint arXiv:1910.03581, 2019. |
| `FedProto` | Y. Tan, G. Long, L. Liu, T. Zhou, Q. Lu, J. Jiang, and C. Zhang,
“Fedproto: Federated prototype learning across heterogeneous clients,”
in Proceedings of the AAAI conference on artificial intelligence, vol. 36,
no. 8, 2022, pp. 8432–8440. |
| `GeFL` | H. Kang, S. Cha, and J. Kang, “Gefl: Model-agnostic federated learning
with generative models,” IEEE Transactions on Mobile Computing,
2025. |

Adding one means dropping a folder under `Algorithm/` and adding a line to
`Algorithm/__init__.py`. Entry points are imported lazily, so a broken or half-finished
algorithm cannot stop the others from running.

## Layout

```
main.py                 Entry point: build data and models, then dispatch
args.py                 Every command-line option, and which of them were typed
Algorithm/
  __init__.py           Name -> entry point registry
  base.py               Shared: hyperparameter precedence, evaluation, metrics CSV
  <Name>/
    main.py             The round loop
    Client.py           Client-side behaviour
    default_args.py     Hyperparameters from that algorithm's paper
TargetModels/           Architectures, all exposing forward/embed/forward_to_mid
datasets/               Dirichlet split, loaders, and the Digit5 domain-shift setup
utils/                  Training loops, client sampling, synthetic-data generation
```

### Hyperparameter precedence

Command line beats an algorithm's `default_args.py`, which in turn beats the generic
defaults in `args.py`. So `--lr 0.05` is always honoured, while an algorithm's published
learning rate still overrides the repository-wide default when the user says nothing.
The resolved values are printed at the start of each run and therefore land in the log.

## Output

Each run writes to `result/<algorithm>/<dataset>/<noniid_alpha>/<timestamp>.*`:

- `.log` — the full transcript, including the resolved hyperparameters
- `_metrics.csv` — one row per (round, client) with the test accuracy, loss, and that
  client's architecture, plus a `client_id = -1` row holding the round average

The CSV exists so that figures do not have to be produced by re-parsing the log:

```python
import pandas as pd
df = pd.read_csv('result/DLSD/CIFAR10/0.1/20260825_120000_metrics.csv')
df[df.client_id == -1].plot(x='round', y='test_acc')          # global accuracy
df[df.client_id >= 0].groupby('model').test_acc.mean()        # per-architecture
```
