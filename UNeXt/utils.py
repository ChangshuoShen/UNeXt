import argparse
import torch.nn as nn

class qkv_transform(nn.Conv1d):
    """
        Conv1d for qkv_transform
        用于在注意力机制中转换查询，键和值的Conv1d层
    """

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    '''计算模型中的可训练参数数量'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """
        Computes and stores the average and current value
        计算和存储平均值和当前值三
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


