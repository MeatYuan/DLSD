import math

import numpy as np

from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.DFML.Client import Client
from Algorithm.DFML.default_args import apply_dfml_defaults
from utils.network import Sample_Client


class CyclicAlpha:
    """Cyclic alpha (DFML Eq 4 / A.2): a sawtooth, not a full oscillation -- alpha cosine-
       anneals to alpha_max on a cycle's last round, then restarts from the bottom. Read off
       Figure 11's vector path (peaks at 10, 21, 33 ... 483, so period 10 growing by 1)."""

    def __init__(self, alpha_min, alpha_max, init_period,
                 mode='linear', period_increment=1, period_mult=2):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.period = init_period
        self.mode = mode
        self.period_increment = period_increment
        self.period_mult = period_mult
        self.pos = 0  # position within the current cycle, in [1, period]
        # position that produced the alpha most recently returned, for rounds_to_peak()
        self._last_pos, self._last_period = 0, init_period

    def _grow_period(self):
        if self.mode == 'linear':
            self.period += self.period_increment
        elif self.mode == 'mult':
            self.period = int(self.period * self.period_mult)
        # 'fixed': leave self.period unchanged

    def step(self):
        self.pos += 1  # pos runs 1..period, so a cycle is exactly `period` rounds
        frac = self.pos / self.period
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 - math.cos(math.pi * frac))

        self._last_pos, self._last_period = self.pos, self.period

        if self.pos >= self.period:  # this round is the maximum; close the cycle
            self.pos = 0
            self._grow_period()
        return alpha

    def rounds_to_peak(self):
        """Rounds between the alpha just returned and this cycle's maximum.
           0 means the last step() produced the maximum itself."""
        return self._last_period - self._last_pos


def _peak_updates(args):
    """How many of a cycle's largest alphas refresh the peak model (A.4.6, participation
       under 50%). The paper's N/|S| contradicts its own example (10% -> "largest five"),
       so this uses the linear scale fitting both anchors: 10%->5, 20%->4 ... >=50%->1."""
    if getattr(args, 'dfml_peak_updates', None) is not None:
        return max(1, int(args.dfml_peak_updates))
    return max(1, round((0.5 - args.sample_rate) * 10) + 1)


def DFML(Client_Data, models, model_list, args):
    apply_dfml_defaults(args)
    recorder = MetricsRecorder(args)

    clients = [Client(model=models[i],
                      local_train_loader=Client_Data["client_train"][i],
                      local_test_loader=Client_Data["client_test"],
                      client_id=i,
                      device=args.device,
                      args=args) for i in range(args.num_clients)]

    scheduler = CyclicAlpha(args.dfml_alpha_min, args.dfml_alpha_max,
                            args.dfml_init_period,
                            mode=args.dfml_period_mode,
                            period_increment=args.dfml_period_increment,
                            period_mult=args.dfml_period_mult)

    print(f"[DFML] teacher KL weighting: {getattr(args, 'dfml_kl_weighting', 'phi')}"
          f"  ({'Eq 3, weighted by trainable parameters' if getattr(args, 'dfml_kl_weighting', 'phi') == 'phi' else 'Appendix A.4.7, vanilla average'})")

    aggregator = 0  # the lowest-id client aggregates first (Algorithm 1)
    rng = np.random.default_rng(seed=57)

    for Round in range(args.Round):
        alpha = scheduler.step()

        # Senders: sampled clients excluding the aggregator. Participants = senders + aggregator.
        sampled = Sample_Client(args.num_clients, sample_rate=args.sample_rate)
        if Round == 0:
            aggregator = min(sampled) 
        else:
            aggregator = rng.choice(sampled) 

        senders = [int(i) for i in sampled if int(i) != aggregator]

        n_peak = _peak_updates(args)
        in_peak_window = scheduler.rounds_to_peak() < n_peak

        peak_tag = f" [peak-window n={n_peak}]" if in_peak_window else ""
        print(f"Round {Round + 1}: Aggregator {aggregator}, Senders {senders}, alpha={alpha:.4f}{peak_tag}")

        # Client side: each participant trains locally on its own data.
        total_loss, total_acc = 0.0, 0.0
        for i in sampled:
            loss, acc = clients[i].local_train()
            total_loss += loss
            total_acc += acc
            print(f'Round {Round + 1}: Client {i} - Train Acc: {acc:.4f}, Loss: {loss:.4f}')
        print(f'Round {Round + 1}: Avg - Train Acc: {total_acc/len(sampled):.4f}, Loss: {total_loss/len(sampled):.4f}')

        # Aggregator side: mutual learning, then update peak models.
        participant_clients = [clients[i] for i in sampled]
        clients[aggregator].mutual_learning(participant_clients, alpha, args)
        for i in sampled:
            clients[i].update_peak(alpha, in_peak_window)

        # Global accuracy: evaluate every client's peak model on the shared test set.
        result = evaluate_all(clients, Client_Data, args, Round + 1,
                              recorder=recorder, model_list=model_list)

    return result
