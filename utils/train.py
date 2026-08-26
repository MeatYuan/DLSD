import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
def Train_model(model: nn.Module, trainloader: DataLoader, loss_fn = nn.CrossEntropyLoss(), optimizer = None, 
                epochs: int = 1, device: str ='cpu', release = True, scheduler = None):
    """Train the model on the training dataset.
        returns average loss, average accuracy
    """
    if device == 'cpu':
        print("Training on CPU")
    if optimizer is None:
        print("No optimizer provided, using SGD with lr=0.01")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model = model.to(device)
    onload_optimizer_states(device, optimizer)
    model.train()

    avg_loss = 0
    avg_acc = 0
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            
            outputs = model(inputs)['logits']
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)
        if scheduler is not None:
            scheduler.step()
        avg_loss += total_loss / total_samples
        avg_acc += total_correct / total_samples

    avg_loss /= epochs
    avg_acc /= epochs

    if release:
        offload_optimizer_states(optimizer)
        model.to('cpu')
        model.eval()
        torch.cuda.empty_cache()

    return avg_loss, avg_acc

def Evaluate(model: nn.Module, testloader: DataLoader, loss_fn = nn.CrossEntropyLoss(), 
             device: str ='cpu', release = True):
    """Evaluate the model on the test dataset.
       returns loss, accuracy
    """
    if device == 'cpu':
        print("Evaluating on CPU")
    
    model = model.to(device)
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)['logits']
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if release:
        model.to('cpu')
        torch.cuda.empty_cache()

    return avg_loss, avg_acc

def Evaluate_with_adapter(model: nn.Module,  adapter: nn.Module, testloader: DataLoader, loss_fn = nn.CrossEntropyLoss(), 
                         device: str ='cpu', release = True):
    if device == 'cpu':
        print("Evaluating on CPU")
    
    model = model.to(device)
    adapter = adapter.to(device)
    model.eval()
    adapter.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward_to_mid(inputs)
            feature = adapter(output)[0]
            outputs = model.forward_from_mid(feature)['logits']
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if release:
        model.to('cpu')
        adapter.to('cpu')
        torch.cuda.empty_cache()

    return avg_loss, avg_acc

        


def get_global_optimizer_schedule(model, adapter, args):
    # for param in list(model.features.parameters()) + list(model.features_sec.parameters()):
    #     param.requires_grad = False
    # for param in list(adapter.encoder.parameters()) + list(adapter.fc_encoder.parameters()):
    #     param.requires_grad = False
    
    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(filter(lambda p: p.requires_grad, adapter.parameters()))
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Round, eta_min=1e-5)

    # for param in list(model.features.parameters()) + list(model.features_sec.parameters()):
    #     param.requires_grad = True
    # for param in list(adapter.encoder.parameters()) + list(adapter.fc_encoder.parameters()):
    #     param.requires_grad = True
        
    return optimizer, scheduler

def offload_optimizer_states(*optimizers):
    """Move optimizer states from GPU to CPU to free VRAM."""
    for opt in optimizers:
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

def onload_optimizer_states(device, *optimizers):
    """Move optimizer states from CPU back to device before training."""
    for opt in optimizers:
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

def KL_divergence(teacher_logits, student_logits, T=1, reduction='batchmean'):
    student_logits = F.log_softmax(student_logits / T, dim=1)
    teacher_logits = F.softmax(teacher_logits / T, dim=1)

    # KL = F.kl_div(student_logits, teacher_logits, reduction='batchmean')*(T**2)
    if reduction == 'none':
        KL = F.kl_div(student_logits, teacher_logits, reduction='none').sum(dim=1)*(T**2)
    else:
        KL = F.kl_div(student_logits, teacher_logits, reduction=reduction)*(T**2)

    return KL

def WSM_loss(logits, labels, class_dist, eps=1e-12):
    """Re-weighted SoftMax cross-entropy (Legate et al., 2023).
       Equation 2 in DFML: L = mean_x [ log sum_c beta_c * exp(z_c) - z_y ].
       class_dist is the per-class proportion (beta) of the data being trained on.
       Classes absent from the data get ~zero weight, protecting against forgetting.
    """
    log_beta = torch.log(class_dist.to(logits.device) + eps)
    lse = torch.logsumexp(logits + log_beta, dim=1)
    z_y = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    return (lse - z_y).mean()
