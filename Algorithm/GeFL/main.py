import copy
import torch

from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.GeFL.Client import Client
from Algorithm.GeFL.DCGAN import Generator, Discriminator
from Algorithm.GeFL.default_args import apply_gefl_defaults
from utils.network import Sample_Client


def GeFL(Client_Data, models, model_list, args):
    """
    GeFL: Generative Federated Learning.
    Reference: Kang et al., IEEE TMC 2025.
    Code: github.com/honggkang/hetero-model-fl-gen (GeFL_DCGAN.py)

    Flow:
      1. GAN warmup (gen_wu_rounds): all selected clients train GAN, FedAvg G+D.
      2. Main rounds:
         a. Each selected client trains GAN locally.
         b. Server FedAvg G and D.
         c. Each selected client trains target model (synthetic → real).
         d. Server grouped-FedAvg target models by architecture.

    Per-client optimizer states (optgs/optds) persisted across rounds, matching official.
    """
    apply_gefl_defaults(args)
    recorder = MetricsRecorder(args)

    global_generator = Generator(
        output_channel=args.image_channel,
        d=256,
    )
    global_discriminator = Discriminator(
        img_channels=args.image_channel,
        d=64,
    )
    global_generator.weight_init(mean=0.0, std=0.02)
    global_discriminator.weight_init(mean=0.0, std=0.02)

    # per-client GAN optimizer states (None until first use)
    optgs = [None] * args.num_clients
    optds = [None] * args.num_clients

    client_list = [
        Client(
            model=models[i],
            local_train_loader=Client_Data['client_train'][i],
            local_test_loader=Client_Data['client_test'],
            client_id=i,
            args=args,
            device=args.device,
        )
        for i in range(args.num_clients)
    ]

    # ── GAN warmup rounds (no target model training) ──────────────────────────
    if args.gen_wu_rounds > 0:
        print(f"\n===== GeFL GAN warmup ({args.gen_wu_rounds} rounds) =====")
        for wu_iter in range(args.gen_wu_rounds):
            g_states, d_states = [], []
            wu_g_losses, wu_d_losses = [], []
            for client in client_list:
                idx = client.get_id()
                g_s, d_s, optgs[idx], optds[idx], avg_g, avg_d = client.train_gan(
                    copy.deepcopy(global_generator),
                    copy.deepcopy(global_discriminator),
                    gan_epochs=args.gan_epochs,
                    optg_state=optgs[idx],
                    optd_state=optds[idx],
                )
                g_states.append(g_s)
                d_states.append(d_s)
                wu_g_losses.append(avg_g)
                wu_d_losses.append(avg_d)
            global_generator = _fedavg_model(global_generator, g_states)
            global_discriminator = _fedavg_model(global_discriminator, d_states)
            print(f"[GAN Warmup {wu_iter:3d}] G_loss: {sum(wu_g_losses)/len(wu_g_losses):.4f} | D_loss: {sum(wu_d_losses)/len(wu_d_losses):.4f}")

    # ── main federated rounds ─────────────────────────────────────────────────
    for glob_iter in range(1,args.Round+1):
        print(f"\n\n------------- Round number: {glob_iter} -------------\n\n")

        sampled_clients_idx = Sample_Client(args.num_clients, sample_rate=args.sample_rate)
        selected_clients = [client_list[i] for i in sampled_clients_idx]
        selected_model_list = [model_list[i] for i in sampled_clients_idx]

        # ── 1. each client trains target model then GAN (matches official order) ─
        g_states, d_states = [], []
        round_g_losses, round_d_losses = [], []
        for client in selected_clients:
            idx = client.get_id()
            client.train_target(copy.deepcopy(global_generator))
            g_s, d_s, optgs[idx], optds[idx], avg_g, avg_d = client.train_gan(
                copy.deepcopy(global_generator),
                copy.deepcopy(global_discriminator),
                gan_epochs=args.gan_epochs,
                optg_state=optgs[idx],
                optd_state=optds[idx],
            )
            g_states.append(g_s)
            d_states.append(d_s)
            round_g_losses.append(avg_g)
            round_d_losses.append(avg_d)

        # ── 2. server: FedAvg G and D ─────────────────────────────────────────
        global_generator = _fedavg_model(global_generator, g_states)
        global_discriminator = _fedavg_model(global_discriminator, d_states)
        print(f"[Round {glob_iter:3d} GAN] G_loss: {sum(round_g_losses)/len(round_g_losses):.4f} | D_loss: {sum(round_d_losses)/len(round_d_losses):.4f}")

        # ── 3. server: grouped FedAvg on target models ────────────────────────
        _aggregate_target_models(client_list, selected_clients, selected_model_list, model_list)

        result = evaluate_all(client_list, Client_Data, args, glob_iter,
                              recorder=recorder, model_list=model_list)

    return result


def _fedavg_model(global_model, local_state_dicts):
    """Unweighted FedAvg over local state_dicts into global_model (matches official FedAvg)."""
    agg = copy.deepcopy(local_state_dicts[0])
    for k in agg.keys():
        for i in range(1, len(local_state_dicts)):
            agg[k] += local_state_dicts[i][k]
        agg[k] = torch.div(agg[k], len(local_state_dicts))
    global_model.load_state_dict(agg)
    return global_model


def _aggregate_target_models(all_clients, selected_clients, selected_model_names, all_model_names):
    """Grouped FedAvg on target models by architecture (model_wise_FedAvg in official)."""
    groups = {}
    for client, model_name in zip(selected_clients, selected_model_names):
        state, size = client.get_model_state()
        if model_name not in groups:
            groups[model_name] = {'states': [], 'sizes': []}
        groups[model_name]['states'].append(state)
        groups[model_name]['sizes'].append(size)

    aggregated = {}
    for model_name, g in groups.items():
        agg = copy.deepcopy(g['states'][0])
        for k in agg.keys():
            for i in range(1, len(g['states'])):
                agg[k] += g['states'][i][k]
            agg[k] = torch.div(agg[k], len(g['states']))
        aggregated[model_name] = agg

    for client, model_name in zip(all_clients, all_model_names):
        if model_name in aggregated:
            client.set_model_state(aggregated[model_name])


