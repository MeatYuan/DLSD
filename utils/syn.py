import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import random
from torch.utils.data import TensorDataset
from TargetModels.main import build_model
#From https://github.com/ubc-tea/DESA/blob/main/iterative_desa.py#L463

def Generate_Client_Syn_Data(Client_dataset_loader, args):
    train_datasets = []
    for i in range(len(Client_dataset_loader)):
        train_datasets.append(Client_dataset_loader[i].dataset)

    #syn data per client
    IPC = args.IPC

    
    label_syns_tmp = torch.tensor(np.array([np.ones(IPC)*i for i in range(args.num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    image_syns = [torch.randn(size=(args.num_classes*IPC, args.image_channel, args.image_size[0], args.image_size[1]), dtype=torch.float, requires_grad=True, device=args.device) for idx in range(args.num_clients)]
    label_syns = [copy.deepcopy(label_syns_tmp).to(args.device) for idx in range(args.num_clients)]

    print('Start distribution matching...')
    for client_idx in range(args.num_clients):
        # organize the real dataset
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(args.num_classes)]
        images_all = [torch.unsqueeze(train_datasets[client_idx][i][0], dim=0) for i in range(len(train_datasets[client_idx]))]
        labels_all = [train_datasets[client_idx][i][1] for i in range(len(train_datasets[client_idx]))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=1, momentum=0.5) # optimizer_img for synthetic data
        inv_iters = args.syn_iters
        image_batch = 256

        for it in range(inv_iters):
            loss_avg = 0
            # get real images for each class
            image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(args.num_classes)]
            # print([image_real[i].size(0) for i in range(len(image_real))])
            loss, image_syns[client_idx] = distribution_matching(image_real, image_syns[client_idx], optimizer_img, args.image_channel, args.num_classes, args.image_size, IPC, args.device, args.random_seed, args=args)
            # report averaged loss
            loss_avg += loss
            loss_avg /= args.num_classes
            if it%100 == 0:
                print(' Initialization:\t client = %2d, iter = %05d, loss = %.4f' % (client_idx, it, loss_avg))
    
    global_virtual_images = [copy.deepcopy(image_syns[client_idx].detach().cpu()).to(args.device) for client_idx in range(args.num_clients)]
    global_virtual_labels = [copy.deepcopy(label_syns[client_idx].detach().cpu()).to(args.device) for client_idx in range(args.num_clients)]

    mixup_virtual_images = torch.mean(torch.stack(global_virtual_images), dim=0).detach().cpu()
    mixup_virtual_labels = global_virtual_labels[0].detach().cpu()

    mixup_train_set = TensorDataset(mixup_virtual_images, mixup_virtual_labels)
    shuffled_idx = list(range(0, len(mixup_train_set)))
    
    random.shuffle(shuffled_idx)
    shuffled_mixup_train_set = torch.utils.data.Subset(mixup_train_set, shuffled_idx[:len(mixup_train_set)])
    kd_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    reg_train_loader = torch.utils.data.DataLoader(shuffled_mixup_train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    concated_train_sets = [torch.utils.data.ConcatDataset([train_dataset, mixup_train_set]) for train_dataset in train_datasets]
    concated_train_loaders = [torch.utils.data.DataLoader(client_concated_train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=safe_collate, drop_last=True) for client_concated_train_set in concated_train_sets]

    return concated_train_loaders, kd_train_loader, reg_train_loader

def get_images(images_all, indices_class, c, n): # get random n images from class c
    if n < len(indices_class[c]):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
    else:
        idx_shuffle_0 = np.random.permutation(indices_class[c])
        idx_shuffle_1 = np.random.permutation(indices_class[c])[:n-len(indices_class[c])]
        idx_shuffle = np.concatenate([idx_shuffle_0, idx_shuffle_1], axis=0)
    return images_all[idx_shuffle]

def distribution_matching(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, device, random_seed, image_server=None, net=None, label_selected=None, args=None):

    lambda_sim = 0.5

    # default we use ConvNet
    if net == None:
        net = build_model('ConvNet', args).to(device) # get a random model
        net.train()
        # for param in list(net.parameters()):
        #     param.requires_grad = False
    else:
        net = net.to(device)
        net.train()
        # for param in list(net.parameters()):
        #     param.requires_grad = False

    embed = net.embed # for GPU parallel

    loss_avg = 0

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    if label_selected is not None:
        iter_class = label_selected
    else:
        iter_class = num_classes
    for c in range(iter_class):
        img_real = image_real[c]
        if img_real.size(0) == 0:
            continue
        img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        
        #seed = int(time.time() * 1000) % 100000
        if not hasattr(distribution_matching, "g"):
            distribution_matching.g = torch.Generator().manual_seed(random_seed)
        seed = torch.randint(0, 100000, (1,), generator=distribution_matching.g).item()
        dsa_param = ParamDiffAug()
        img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
        img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)

        output_real = embed(img_real).detach()
        output_syn = embed(img_syn)

        if image_server is not None:
            img_server = image_server[c]
            img_server = DiffAugment(img_server, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
            output_server = embed(img_server).detach()
            server_client_loss = torch.sum((torch.mean(output_server, dim=0) - torch.mean(output_syn, dim=0))**2)
            loss += lambda_sim * server_client_loss
        
        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

    # # l2 and total variation loss
    # loss += lambda_sim * l2_norm(img_syn)
    # loss += lambda_sim * total_variation(img_syn)
            
    optimizer_img.zero_grad()
    loss.backward()
    # total_norm = 0.
    # for group in optimizer_img.param_groups:
    #     for param in group['params']:
    #         if param.requires_grad:
    #             total_norm += param.grad.data.norm(2).item() ** 2.
    # total_norm = total_norm ** .5
    optimizer_img.step()

    if image_server is not None:
        return loss.item(), image_syn, server_client_loss.item()
    else:
        return loss.item(), image_syn#, total_norm

    
class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x

# Auther implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese and theta.size(0) != 1: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese and theta.size(0) != 1: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese and randf.size(0) != 1: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese and randb.size(0) != 1:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese and rands.size(0) != 1:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese and randc.size(0) != 1:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese and translation_x.size(0) != 1:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij" 
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese and offset_x.size(0) != 1:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij" 
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        #print(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            #print(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class Distance_loss(nn.Module):
    def __init__(self, distance="SupCon", device=None):
        super(Distance_loss, self).__init__()
        self.distance = distance
        self.device = device
        if self.distance == "SupCon":
            self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
        else:
            self.supcon_loss = None


    def forward(self, x1, x2, label1=None, label2=None):
        if self.distance == "L2_norm":
            loss = self.L2_norm(x1, x2)
        elif self.distance == "cosine":
            loss = self.cosine(x1, x2)
        elif self.distance == "SupCon":
            loss = self.supcon(x1, x2, label1, label2)
        else:
            raise NotImplementedError
        return loss


    def L2_norm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    def cosine(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos.mean()
        return loss

    def supcon(self, feature1, feature2, label1, label2):

        all_features = torch.cat([feature1, feature2], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([label1, label2], dim=0),
            temperature=0.07, mask=None)
        return align_cls_loss
    
def calculate_kd_loss(y_pred_student, y_pred_teacher, y_true, loss_fn, temp=20., distil_weight=0.9):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / temp, dim=1)
        soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)

        loss = (1. - distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (distil_weight * temp * temp) * loss_fn(
            soft_student_out, soft_teacher_out
        )
        

        return loss

def safe_collate(batch):
    xs, ys = zip(*batch)  # [(x,y), ...]
    xs = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in xs]
    ys = [y if isinstance(y, torch.Tensor) else torch.tensor(y)    for y in ys]
    return torch.stack(xs, 0).float(), torch.stack(ys, 0).long()