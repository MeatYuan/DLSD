import torch
import torch.nn as nn
import numpy as np
import copy
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from datasets.loader import SynDataset, ClassBatchSampler
from datasets.Data import Dataset_Mean_Std
from utils.train import Train_model, Evaluate, Evaluate_with_adapter, get_global_optimizer_schedule, KL_divergence
from TargetModels.Adapter import AdapterMLP, AdapterConv
from TargetModels.main import get_model
from utils.syn import get_images, distribution_matching
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from utils.train import onload_optimizer_states, offload_optimizer_states
class Client:
    def __init__(self, model, local_train_loader, local_test_loader, client_id, args, local_val_loader = None,device = "gpu"):
        # client id
        self.client_id = client_id
        self.args = args

        # local model
        self.local_model = model
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.Round, eta_min=1e-5)
        # data loader
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader
        self.local_iter = iter(self.train_loader)
        self.syn_dataset = None
        self.class_loader = None

        # neighbor clients
        self.neighbors = defaultdict(Neighbor)
        self.global_dataset = {}

        self.device = device
        self.print_info()

        if args.with_adapter:
            self.init_adapter(args)
        else:
            self.adapter = None
        
    def get_id(self):
        """Return the client ID."""
        return self.client_id

    def print_info(self):
        """Print the client information."""
        print("============================================")
        print(f"Client ID: {self.client_id}")
        print(f"Local Model: {self.local_model.__class__.__name__}")
        print(f"Train Loader: {len(self.train_loader.dataset)}")
        print(f"Test Loader: {len(self.test_loader.dataset)}")
        print(f"Device: {self.device}")
        print("============================================")
    
    def Init_neighbors(self, network_info):
        if len(network_info) == 0:
            print(f"Client {self.client_id} has no neighbor.")
            return

        for k in network_info.keys():
            self.neighbors[k] = Neighbor(network_info[k])
        
        print(f"Client {self.client_id} Neighbors : " + ", ".join(str(int(k)) for k in self.neighbors.keys()))
    
    def evaluate(self):
        args = self.args
        self.reset()
        if args.Evaluate_with_adapter:
            loss, acc = Evaluate_with_adapter(self.local_model, self.adapter, self.test_loader,device = self.device)
        else:
            loss, acc = Evaluate(self.local_model, self.test_loader,device = self.device)
        return loss, acc
    
    def init_adapter(self,args):
        self.local_model = self.local_model.to(self.device)
        self.local_model.eval()
        
        dummy_input = torch.randn(2, args.image_channel, args.image_size[0], args.image_size[1]).to(self.device) 
        with torch.no_grad():
            output = self.local_model.forward_to_mid(dummy_input)
            input_shape = output.shape[1:]

        if args.adapter_type == 'conv':
            self.adapter = AdapterConv().to(self.device)
        elif args.adapter_type == 'mlp':
            self.adapter = AdapterMLP(input_shape = input_shape).to(self.device)
        else:
            raise ValueError(f"Unknown adapter type: {args.adapter_type}")

        self.adapter.eval()

        with torch.no_grad():
            ada_output, _ = self.adapter(output)

        print(f"Client {self.client_id} Adapter Input shape: {output.shape}, Output shape: {ada_output.shape}")
        self.adapter_optimizer = torch.optim.SGD(self.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        self.global_optimizer, self.global_scheduler = get_global_optimizer_schedule(self.local_model, self.adapter, args)
        
        adapter_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"Client {self.client_id} Adapter Parameters: {adapter_params / 1e6:.4f} M")
    
    def Generate_Syn_Data(self, args):
        #每個labels要生成的數量
        IPC = args.IPC 
        self.syndata_info = SynData_info(self.train_loader, args)
        if len(self.syndata_info.labels_selected) == 0:
            print(f"Client {self.client_id} has no labels with enough samples for synthetic data generation.")
            return
        self.image_syns = torch.randn(size=(len(self.syndata_info.labels_selected)*IPC, args.image_channel, args.image_size[0], args.image_size[1]), dtype=torch.float, device=self.device, requires_grad=True)
        self.label_syns = torch.tensor(np.array([np.ones(IPC)*label for label in self.syndata_info.labels_selected]), dtype=torch.long, requires_grad=False, device=self.device).view(-1) 
        self.optimizer_img = torch.optim.SGD([self.image_syns], lr=args.syn_lr, momentum=args.syn_momentum)
        print(f"Client {self.client_id} Generate Synthetic Data for labels: {self.syndata_info.labels_selected} with {IPC} images each.")


    def iterate_syn_data(self, args, rounds):
        if len(self.syndata_info.labels_selected) == 0:
            print(f"Client {self.client_id} has no labels with enough samples for synthetic data generation.")
            return
        images_all = self.syndata_info.images_all.to(self.device)
        self.image_syns = self.image_syns.to(self.device)
        mean = torch.tensor(self.syndata_info.mean).view(1, args.image_channel, 1, 1).to(self.device)
        std = torch.tensor(self.syndata_info.std).view(1, args.image_channel, 1, 1).to(self.device)
        onload_optimizer_states(self.device, self.optimizer_img)
        for it in range(args.syn_iters):
            image_real = [get_images(images_all, self.syndata_info.indices_class, label, args.syn_batch_size) for label in self.syndata_info.labels_selected]
            loss, self.image_syns = distribution_matching(image_real, self.image_syns, self.optimizer_img, args.image_channel, args.num_classes, args.image_size, args.IPC, self.device, args.random_seed, label_selected=len(self.syndata_info.labels_selected), net=copy.deepcopy(self.local_model))
            with torch.no_grad():
                self.image_syns.data = self.image_syns.data * std + mean
                            
                # 2. Clip: 強制限制在 [0, 1]
                self.image_syns.data.clamp_(0, 1)
                            
                # 3. Normalize: 再轉回 [-2.4, 2.7] 範圍
                self.image_syns.data = (self.image_syns.data - mean) / std
            loss = loss / len(self.syndata_info.labels_selected)
            if it % 100 == 0:
                print(f"Client {self.client_id} Synthetic Data Generation Iteration: {it}, Loss: {loss}")
        
        virtual_images = copy.deepcopy(self.image_syns.detach().cpu())
        virtual_labels = copy.deepcopy(self.label_syns.detach().cpu().tolist())
        self.syn_dataset = TensorDataset(virtual_images, virtual_labels)
        print(f"Client {self.client_id} Synthetic Data Generation Completed. Size: {len(self.syn_dataset)}")
        
        # # 只存最後一次生成的影像：每張圖單獨存、依類別分資料夾，固定檔名每次覆蓋
        # # 反正規化回 [0,1] 以便正常顯示
        # vis_images = (virtual_images * std.cpu() + mean.cpu()).clamp(0, 1)
        # per_class_idx = defaultdict(int)
        # for img, label in zip(vis_images, virtual_labels):
        #     class_dir = f"syn_images/{args.dataset}/DLSD/{args.noniid_alpha}/class_{int(label)}"
        #     os.makedirs(class_dir, exist_ok=True)
        #     idx = per_class_idx[int(label)]
        #     save_image(img, f"{class_dir}/client_{self.client_id}_{idx}.png")
        #     per_class_idx[int(label)] += 1
        # print(f"Client {self.client_id} Synthetic images saved under syn_images/{args.dataset}/DLSD/{args.noniid_alpha}/")

        
        offload_optimizer_states(self.optimizer_img)
        

    def get_features_dataset(self, args, rounds):

        if len(self.syndata_info.labels_selected) == 0:
            print(f"Client {self.client_id} has no labels with enough samples for synthetic data generation.")
            self.global_dataset[self.client_id] = ({}, rounds)
            return
        
        if len(self.global_dataset) == 0 or self.global_dataset[self.client_id][1] // args.syn_round < rounds // args.syn_round:
            print(f"Client {self.client_id} Generate Global Dataset from Synthetic Data at Round {rounds}.")
            self.iterate_syn_data(args, rounds)

        self.local_model = self.local_model.to(self.device)
        self.local_model.eval()
        if self.adapter is not None:
            self.adapter = self.adapter.to(self.device)
            self.adapter.eval()

        images_list = []
        features_list = []
        logits_list = []
        
        syn_loader = DataLoader(self.syn_dataset, batch_size=32, shuffle=False, num_workers=2)
        with torch.no_grad():
            for inputs, labels in syn_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.local_model(inputs)
                if self.adapter is None:
                    features = torch.zeros((inputs.size(0), 1), device=self.device)
                else:
                    _, features = self.adapter(outputs['Z'])

                preds = torch.argmax(outputs['logits'], dim=1)
                mask = (preds == labels)

                if mask.sum() > 0:
                    images_list.append(inputs[mask].detach().cpu())
                    features_list.append(features[mask].detach().cpu())
                    logits_list.append(outputs['logits'][mask].detach().cpu())

        if len(images_list) == 0:
            print(f"Client {self.client_id} has no correct predictions on synthetic data.")
            self.global_dataset[self.client_id] = ({}, rounds)
        else:
            global_dataset = SynDataset(
                images=torch.cat(images_list, dim=0),
                features=torch.cat(features_list, dim=0),
                logits=torch.cat(logits_list, dim=0)
            )
            print(f"Client {self.client_id} Generated Global Dataset from Synthetic Data. Size: {len(global_dataset)}")
            self.global_dataset[self.client_id] = (global_dataset, rounds)
            
        
        self.release()

    def transmit_info(self, adapter_agg = False):
        adapter_weight = None
        datasize = None
        if adapter_agg:
            adapter_weight = {
                'fc_encoder' : self.adapter.fc_encoder.state_dict(),
                'fc_decoder' : self.adapter.fc_decoder.state_dict(),
            }
            datasize = len(self.train_loader.dataset)
        for neighbor_id, neighbor in self.neighbors.items():
            dataset_to_send = {self.client_id: self.global_dataset[self.client_id]}
            neighbor.client.receive_info(dataset_to_send, self.client_id, adapter_weight, datasize)
            # neighbor.client.receive_info(self.global_dataset, self.client_id, adapter_weight, datasize)

    def receive_info(self, neighbor_global_dataset, neightbor_id, adapter_weight = None, datasize = None):
        for client_id, v in neighbor_global_dataset.items():
            if client_id not in self.global_dataset:
                self.global_dataset[client_id] = copy.deepcopy(v)
            else:
                if v[1] > self.global_dataset[client_id][1]:
                    self.global_dataset[client_id] = copy.deepcopy(v)

        if adapter_weight is not None:
            self.neighbors[neightbor_id].adapter_weight = copy.deepcopy(adapter_weight)
            self.neighbors[neightbor_id].datasize = datasize
     
    def aggergate_adapter(self, ):
        if len(self.neighbors) == 0:
            return
        
        weights_sum = len(self.train_loader.dataset)
        agg_weights = {self.client_id: weights_sum}
        for neighbor_id in self.neighbors.keys():
            neighbor_weight = self.neighbors[neighbor_id].datasize
            weights_sum += neighbor_weight
            agg_weights[neighbor_id] = neighbor_weight
        
        modules_to_aggregate = {
                'fc_encoder': self.adapter.fc_encoder,
                'fc_decoder': self.adapter.fc_decoder,
            }

        for module_name, module in modules_to_aggregate.items():
            current_state = module.state_dict()
            aggregated_state = OrderedDict()
            
            for param_name, current_param in current_state.items():
                current_weight = agg_weights[self.client_id] / weights_sum
                
                agg_tensor = torch.mul(current_param.clone(), current_weight)

                # (b) 加入所有 neighbor 的加權參數
                for neighbor_id, neighbor in self.neighbors.items():
                    neighbor_weight = agg_weights[neighbor_id] / weights_sum
                    neighbor_param = neighbor.adapter_weight[module_name][param_name]
                    agg_tensor.add_(torch.mul(neighbor_param, neighbor_weight))
                
                aggregated_state[param_name] = agg_tensor
            
            module.load_state_dict(aggregated_state)
    
    def train_with_global_dataset(self, args):
        if len(self.global_dataset) == 0:
            print(f"Client {self.client_id} has no global dataset.")
            return
        
        # dataset concat
        combined_dataset = self.Get_Global_Dataset()
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        global_iter = iter(combined_loader)
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()
        onload_optimizer_states(self.device, self.optimizer)
        criterion = nn.CrossEntropyLoss()

        if self.adapter is not None:
            self.adapter = self.adapter.to(self.device)
            self.adapter.train()
            onload_optimizer_states(self.device, self.adapter_optimizer)
            criterion_align = nn.CosineEmbeddingLoss()
            target_ones = torch.ones(args.batch_size).to(self.device)
            mse = nn.MSELoss()

        avg_loss = 0
        avg_acc = 0
        avg_kd_loss = 0
        avg_local_loss = 0
        avg_mse_loss = 0
        iii = 0
        avg_ada_loss = 0
        avg_ada_ce_loss = 0
        avg_ada_align_loss = 0
        for epoch in range(args.local_epochs):
            total_loss = 0
            total_acc = 0
            total_size = 0
            total_kd_loss = 0
            total_local_loss = 0
            total_local_size = 0
            total_mse_loss = 0
            total_ada_loss = 0
            total_ada_ce_loss = 0
            total_ada_align_loss = 0
            for local_data, local_target in self.train_loader:
                local_data, local_target = local_data.to(self.device), local_target.to(self.device)

                while True:
                    try:
                        data, global_feature, global_logits = next(global_iter)
                    except StopIteration:
                        global_iter = iter(combined_loader)
                        data, global_feature, global_logits = next(global_iter)
                    if data.size(0) >= 1:
                        break
                data, global_feature, global_logits = data.to(self.device), global_feature.to(self.device), global_logits.to(self.device)
                _, target = torch.max(global_logits, 1)

                if self.adapter is not None:
                    if data.size(0) != target_ones.size(0):
                        target_ones = torch.ones(data.size(0)).to(self.device)
                    self.adapter.train()
                    with torch.no_grad():
                        out_dict = self.local_model(data)
                        feat_detached = out_dict['Z'].detach()


                    noise = torch.randn_like(feat_detached) * 0.1
                    feat_detached = feat_detached + noise
                    x_recon, z_out = self.adapter(feat_detached)
                    loss_align = criterion_align(z_out, global_feature, target_ones)
                    ada_logit_output = self.local_model.forward_from_mid(x_recon)['logits']
                    
                    ada_ce_loss = criterion(ada_logit_output, target)
                    adapt_loss = args.train_adapter_mse_weight * loss_align + args.train_adapter_decoder_weight * ada_ce_loss
                    self.adapter_optimizer.zero_grad()
                    adapt_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=5.0)
                    self.adapter_optimizer.step()
                    total_ada_loss += adapt_loss.item() * data.size(0) 
                    total_ada_ce_loss += ada_ce_loss.item() * data.size(0) 
                    total_ada_align_loss += loss_align.item() * data.size(0)

                if local_data.size(0) <= 1:
                    continue
                output_local = self.local_model(local_data)['logits']

                outputs = self.local_model(data)
                feature = outputs['Z']
                output = outputs['logits']

                kd_loss = KL_divergence(global_logits, output, T = args.Distallation_T)
                loss_local = criterion(output_local, local_target)

                loss = args.global_kl_weight * kd_loss + \
                       args.global_local_weight * loss_local 

                if self.adapter is not None:
                    self.adapter.eval()
                    _, feature = self.adapter.forward(feature)
                    mse_loss = criterion_align(feature, global_feature, target_ones)
                    loss += args.global_mse_weight * mse_loss
                    total_mse_loss += mse_loss.item() * data.size(0)
                else:
                    total_mse_loss = 0


                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += loss.item() * data.size(0) 
                _, pred = torch.max(output, 1)
                total_acc += (pred == target).sum().item()
                total_size += data.size(0)
                total_kd_loss += kd_loss.item() * data.size(0)
                total_local_loss += loss_local.item() * local_data.size(0)
                total_local_size += local_data.size(0)
            # self.global_scheduler.step()
            self.scheduler.step()
            avg_ada_loss += total_ada_loss / total_size
            avg_ada_ce_loss += total_ada_ce_loss / total_size
            avg_ada_align_loss += total_ada_align_loss / total_size
            avg_loss += total_loss / total_size
            avg_acc += total_acc / total_size
            avg_kd_loss += total_kd_loss / total_size
            avg_mse_loss += total_mse_loss / total_size
            avg_local_loss += total_local_loss / total_local_size
        avg_loss /= args.local_epochs
        avg_acc /= args.local_epochs
        avg_kd_loss /= args.local_epochs
        avg_local_loss /= args.local_epochs
        avg_mse_loss /= args.local_epochs
        avg_ada_loss /= args.local_epochs
        avg_ada_ce_loss /= args.local_epochs
        avg_ada_align_loss /= args.local_epochs
        
        self.release()

        return avg_loss, avg_acc, avg_kd_loss, avg_local_loss, avg_mse_loss, avg_ada_loss, avg_ada_ce_loss, avg_ada_align_loss
    
    def Get_Global_Dataset(self, ):
        dataset = [v[0] for client_id, v in self.global_dataset.items() if client_id != self.client_id]
        combined_dataset = ConcatDataset(dataset)
        return combined_dataset

    def reset(self):
        self.neighbors = defaultdict(Neighbor)

    def save_model(self, args):
        path = f"model/DLSD/{args.noniid_alpha}/client_{self.client_id}_model.pth"
        torch.save(self.local_model.state_dict(), path)
        if self.adapter is not None:
            adapter_path = f"model/DLSD/{args.noniid_alpha}/client_{self.client_id}_adapter.pth"
            torch.save(self.adapter.state_dict(), adapter_path)

    def release(self, ):
        self.local_model.to('cpu')
        offload_optimizer_states(self.optimizer)
        if self.adapter is not None:
            self.adapter.to('cpu')
            offload_optimizer_states(self.adapter_optimizer)
        torch.cuda.empty_cache()

class Neighbor:
    def __init__(self, client : Client):
        self.client : Client = client
        self.adapter_weight = None
        self.datasize = 0

class SynData_info:
    def __init__(self, train_loader, args):
        #讀取訓練集中每一個資料的圖片和標籤
        images_all = []
        labels_all = []
        for images, labels in train_loader:
            images_all.append(images)
            labels_all.append(labels)
        self.images_all = torch.cat(images_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0).long()
        
        indices_class = [[] for _ in range(args.num_classes)]
        for idx, label in enumerate(labels_all.cpu().numpy()):
            indices_class[label].append(idx)
        self.indices_class = indices_class
        
        labels_selected = [label for label in range(len(indices_class)) if len(indices_class[label]) >= args.syn_batch_size]
        self.labels_selected = sorted(labels_selected)
        mean,std = Dataset_Mean_Std(args.dataset)
        self.mean = mean
        self.std = std