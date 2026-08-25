from Algorithm.base import MetricsRecorder, evaluate_all
from Algorithm.Local.Client import Client
from utils.network import Sample_Client


def Local(Client_Data, models, model_list, args):
    recorder = MetricsRecorder(args)
    client_list = [Client(model = models[i],
                          local_train_loader = Client_Data["client_train"][i], 
                          local_test_loader = Client_Data["client_test"], 
                          client_id = i,  
                          device=args.device,
                          args = args) for i in range(args.num_clients)]

    for Round in range(args.Round):
        sampled_clients_idx = Sample_Client(len(client_list), sample_rate=args.sample_rate)
        sampled_clients = [client_list[i] for i in sampled_clients_idx]

        total_loss, total_acc = 0,0
        for client in sampled_clients:
            loss, acc = client.train_local_model(args)
            print(f'Round {Round + 1}: Client {client.get_id()} - Train Acc: {acc:.4f}, Loss: {loss:.4f}')
            total_loss += loss
            total_acc += acc
        print(f'Round {Round + 1}: Avg - Train Acc: {total_acc/len(sampled_clients):.4f}, Loss: {total_loss/len(sampled_clients):.4f}')
        
        result = evaluate_all(client_list, Client_Data, args, Round + 1,
                              recorder=recorder, model_list=model_list)

    return result

    
          