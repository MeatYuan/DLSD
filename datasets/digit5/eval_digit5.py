"""
Digit-5 intra/inter-domain evaluation.

For every client, evaluate its model on ALL 5 domain test sets and build a
home-domain x eval-domain accuracy matrix:
  * diagonal  = intra-domain accuracy (client tested on its own domain)
  * off-diag  = inter-domain / cross-domain accuracy (tested on other domains)

Reports the 5x5 matrix + intra avg / inter avg / worst-domain.
Reached through Algorithm.base.evaluate_all() whenever args.dataset == 'Digit5'.
"""
from collections import defaultdict

import numpy as np

from utils.train import Evaluate


def eval_digit5(client_list, domain_test, client_domain, args, round_idx=None):
    """
    client_list  : list of trained clients (each has .local_model)
    domain_test  : {domain_name: DataLoader}  (all 5 domain test sets)
    client_domain: [domain_name] * num_clients (each client's home domain)
    """
    domains = list(domain_test.keys())

    # acc[home][evald] = list of client accuracies
    acc = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(client_list):
        home = client_domain[i]
        for evald in domains:
            _, a = Evaluate(c.local_model, domain_test[evald], device=args.device)
            acc[home][evald].append(a)

    # mean over the clients belonging to each home domain
    mat = {h: {d: (float(np.mean(acc[h][d])) if acc[h][d] else float("nan")) for d in domains}
           for h in domains}

    tag = f"(round {round_idx})" if round_idx is not None else "(final)"
    print(f"\n================ Digit-5 domain accuracy matrix {tag} ================")
    print("home\\eval  " + "".join(f"{d:>12s}" for d in domains))
    for h in domains:
        print(f"{h:<10s}" + "".join(f"{mat[h][d] * 100:>11.2f}%" for d in domains))

    intra = float(np.nanmean([mat[h][h] for h in domains]))
    inter = float(np.nanmean([mat[h][d] for h in domains for d in domains if d != h]))
    worst_home = min(domains, key=lambda h: mat[h][h])
    worst = mat[worst_home][worst_home]
    print("-------------------------------------------------------------------")
    print(f"  intra-domain avg (diagonal)     : {intra * 100:.2f}%")
    print(f"  inter-domain avg (off-diagonal) : {inter * 100:.2f}%")
    print(f"  worst intra-domain              : {worst * 100:.2f}%  ({worst_home})")
    print("===================================================================\n")

    return {"intra": intra, "inter": inter, "worst": worst,
            "worst_domain": worst_home, "matrix": mat}
