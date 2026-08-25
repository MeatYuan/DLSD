import math
import torch.nn as nn
import torch.nn.functional as F
import torch

decay_max = -0.0 
decay_min = -1.0
decay_loops = 5.0

t_start =  1.0
t_end = 20.0

# https://github.com/zhengli97/CTKD/tree/master
class Temp_MLP1(nn.Module):
    def __init__(self, num_classes=100, inter_channel=256, topk_num=10):
        super(Temp_MLP1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * topk_num, inter_channel, 1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, 1, 1)
        )
        self.grl = GradientReversal()

        self.topk_num = topk_num

        for m in self.conv2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight.data, 0)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, teacher_out, student_out, lambda_):
        stu_out_topk = torch.zeros(student_out.shape[0], self.topk_num)

        tea_out = F.softmax(teacher_out, dim=1)
        stu_out = F.softmax(student_out, dim=1)

        tea_out_topk, out_index = tea_out.topk(self.topk_num, dim=1)
        stu_out_topk = stu_out.gather(1, out_index)

        comb_out = torch.cat([tea_out_topk, stu_out_topk], dim=1)
        comb_out = comb_out.view(comb_out.shape[0], comb_out.shape[1], 1, 1).detach()

        T = self.conv1(comb_out)
        T = self.conv2(T)
        T = self.grl(T, lambda_)
        T = T.view(-1)
        
        return T


from torch.autograd import Function
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

class CosineDecay(object):
    def __init__(self,):
        self._max_value = decay_max
        self._min_value = decay_min
        self._num_loops = decay_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = temp.cuda()
        
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s/T, dim=1),
                                F.softmax(y_t/T, dim=1)) * T * T
        
        return KD_loss