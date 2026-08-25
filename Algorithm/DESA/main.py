import torch

from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.DESA.Client import Client
from Algorithm.DESA.default_args import apply_desa_defaults
from utils.network import Sample_Client
from utils.syn import Generate_Client_Syn_Data


def DESA(Client_Data, models, model_list, args):
    apply_desa_defaults(args)
    recorder = MetricsRecorder(args)
    # Get Dataloaders for all clients
    state = torch.get_rng_state()
    Client_train_loader, kd_train_loader, reg_train_loader = Generate_Client_Syn_Data(Client_Data['client_train'], args)
    torch.set_rng_state(state)

    # Initialize clients
    clients = [Client(model=models[i],
                      local_train_loader=Client_train_loader[i],
                      local_test_loader=Client_Data['client_test'],
                      kd_loader=kd_train_loader,
                      reg_loader=reg_train_loader,
                      client_id=i,
                      device=args.device,
                      args=args) for i in range(args.num_clients)]

    # Training process
    for iter in range(args.Round):
        print(f"Global Communication Round {iter+1}/{args.Round}")
        network_info, sampled_clients_idx = Sample_Client(args.num_clients, sample_rate=args.sample_rate, clients = clients)

        # 優先計算KD logits
        for client_id in sampled_clients_idx:
            clients[client_id].calculate_kd_logits()

        for client_id in sampled_clients_idx:
            clients[client_id].Init_neighbors(network_info[clients[client_id].get_id()]['Neighbor'])

        for client_idx in sampled_clients_idx:
            acc, loss_all, loss_ori, loss_kd, loss_reg = clients[client_idx].train_kd_vhl(args=args)   
            print(f"Round {iter+1}: Client {client_idx} - Train Acc: {acc:.4f}, Total Loss: {loss_all:.4f}, CE Loss: {loss_ori:.4f}, KD Loss: {loss_kd:.4f}, Reg Loss: {loss_reg:.4f}")
        
        result = evaluate_all(clients, Client_Data, args, iter + 1,
                              recorder=recorder, model_list=model_list)

    return result






