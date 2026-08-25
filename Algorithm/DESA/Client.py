import torch
import torch.nn as nn
import copy
from utils.syn import calculate_kd_loss, Distance_loss
from utils.train import Train_model, Evaluate
class Client:
    def __init__(self, model, local_train_loader, local_test_loader, kd_loader, reg_loader, client_id, local_val_loader = None,device = "gpu", args=None):
        # client id
        self.client_id = client_id

        # local model
        self.local_model = model

        # data loader
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader
        self.kd_loader = copy.deepcopy(kd_loader)
        self.reg_loader = copy.deepcopy(reg_loader)

        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        self.local_kd_logits = None
        self.neighbor = {}

        self.device = device
        self.print_info()

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
        #print(f"Validation Loader: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("============================================")

    def Pretrain_model(self, args):
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
        classifier_criterion = nn.CrossEntropyLoss()
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()
        for epoch in range(args.local_epochs):
            num_data = 0
            loss_all = 0
            total_correct = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(data)['logits']
    

                loss = classifier_criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                num_data += target.size(0)
                loss_all += loss.item() 

                preds = outputs.argmax(dim=1)
                total_correct += (preds == target).sum().item()
            
            train_loss = loss_all/len(self.train_loader)
            train_acc = total_correct/num_data
            

            if (epoch+1) % 10 == 0:
                test_acc, test_loss = self.evaluate()
                print(f"Train | Client {self.client_id} - Epoch {epoch+1}/{args.local_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"Test  | Client {self.client_id} - Epoch {epoch+1}/{args.local_epochs}, Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        self.local_model.eval()
        self.local_model = self.local_model.to('cpu')

        return {'Train Loss': train_loss, 'Train Acc': train_acc, 'Test Loss': test_loss, 'Test Acc': test_acc}
     

    def evaluate(self,):
        loss, acc = Evaluate(self.local_model, self.test_loader, device = self.device)
        return loss, acc
    
    def calculate_kd_logits(self,):
        self.local_model = self.local_model.to(self.device)
        self.local_model.eval()
        self.local_kd_logits = []
        with torch.no_grad():
            for data, _ in self.kd_loader:
                data = data.to(self.device)
                outputs = self.local_model(data)['logits']
                self.local_kd_logits.append(outputs.detach().cpu())
    
    def Init_neighbors(self, neighbor):
        if len(neighbor) == 0:
            print(f"Client {self.client_id} has no neighbor.")
            return
        
        self.neighbor = neighbor
        print(f"Client {self.client_id} Neighbors : ",", ".join(str(int(k)) for k in neighbor.keys()))

    def upload_kd_logits(self):
        return self.local_kd_logits
    
    def train_kd_vhl(self, lambda_ori=1., lambda_kd=1., lambda_reg=1., args=None):
        self.local_model.train()
        self.local_model = self.local_model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        kd_loss = nn.KLDivLoss()

        distance_loss = Distance_loss(device=self.device)

        example_logits = []
        for idx, neighbor in self.neighbor.items():
            example_logits.append(neighbor.upload_kd_logits())

        avg_correct = 0
        avg_loss_all = 0
        avg_loss_ori_all = 0
        avg_loss_kd_all = 0
        avg_loss_reg_all = 0
        for epoch in range(args.local_epochs):
            correct = 0
            loss_all = 0
            loss_ori_all = 0
            loss_kd_all = 0
            loss_reg_all = 0
            train_iter = iter(self.train_loader)
            kd_iter = iter(self.kd_loader)
            kd_step = 0
            reg_iter = iter(self.reg_loader)
            for step in range(len(train_iter)):
                
                # get classification loss
                x, y = next(train_iter)
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                outputs = self.local_model(x)

                output = outputs['logits']
                features = outputs['feature']
                loss_ori = criterion(output, y)
                loss = loss_ori


                # get kd loss
                if len(example_logits)!= 0:
                    try:
                        x_kd, y_kd = next(kd_iter)
                    except:
                        kd_iter = iter(self.kd_loader)
                        x_kd, y_kd = next(kd_iter)
                        kd_step = 0 # to make sure we get the correct logits from other clients
                    x_kd = x_kd.to(self.device).float()
                    y_kd = y_kd.to(self.device).long()

                    kd_output = self.local_model(x_kd)['logits']
                    # loss = loss_fun(output, output_target)
                    teacher_logits = []
                    for logits in example_logits:
                        teacher_logits.append(logits[kd_step])
                    teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0).to(self.device)
                    loss_kd = calculate_kd_loss(kd_output, teacher_logits, y_kd, kd_loss)
                    kd_step += 1
                    loss_kd_all += loss_kd.item()
                else:
                    loss_kd = 0


                # get regularization loss
                try:
                    x_reg, y_reg = next(reg_iter)
                except:
                    reg_iter = iter(self.reg_loader)
                    x_reg, y_reg = next(reg_iter)
                x_reg = x_reg.to(self.device).float()
                y_reg = y_reg.to(self.device).long()
                reg_feature = self.local_model.embed(x_reg).detach()
                loss_reg = distance_loss(features, reg_feature, y, y_reg) # sup contrastive

                loss = args.lambda_ori * loss_ori + \
                       args.lambda_kd * loss_kd + \
                       args.lambda_reg * loss_reg
                
                

                self.optimizer.zero_grad()
                loss.backward()
                loss_all += loss.item()
                loss_ori_all += loss_ori.item()
                loss_reg_all += loss_reg.item()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
            avg_correct += correct/len(self.train_loader.dataset)
            avg_loss_all += loss_all/len(train_iter)
            avg_loss_ori_all += loss_ori_all/len(train_iter)
            avg_loss_kd_all += loss_kd_all/len(train_iter)
            avg_loss_reg_all += loss_reg_all/len(train_iter)
        
        avg_correct /= args.local_epochs
        avg_loss_all /= args.local_epochs
        avg_loss_ori_all /= args.local_epochs
        avg_loss_kd_all /= args.local_epochs
        avg_loss_reg_all /= args.local_epochs
        return avg_correct, avg_loss_all, avg_loss_ori_all, avg_loss_kd_all, avg_loss_reg_all

    def save_model(self):
        path = f"model/DESA/client_{self.client_id}_model.pth"
        torch.save(self.local_model.state_dict(), path)


    

