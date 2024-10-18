import torch
import torch.nn as nn
import torch.nn.functional as F
# 前三个import我很好理解，下面这个暂时不知道，GitHub下载后发现是jupyter文件，后续再看吧，反正不影响程序运行
'''
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass
'''
# Lovász-Softmax 损失：神经网络中交并集测度优化的可处理替代项

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']    
# 定义了当其他模块使用 from module import * 语句导入这个模块时，只有 BCEDiceLoss 和 LovaszHingeLoss 这两个名字会被导入

# 二元交叉熵损失（Binary Cross Entropy，BCE）和Dice损失
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类 nn.Module 的构造函数，(PyTorch的规定，如果我们覆盖了构造函数，则需要在其中调用父类的构造函数)

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target) # 计算二元交叉熵损失（BCE）
        smooth = 1e-5                                           # 定义一个平滑值，用于Dice损失的计算，防止分母为0
        input = torch.sigmoid(input)                            # 输入到sigmoid中，转化到（0,1）
        num = target.size(0)                                    # 获得目标批次(batch)大小
        input = input.view(num, -1)                             # 将输入调整为二维张量，第一维是批次大小，第二维是其他维度
        target = target.view(num, -1)                           # 将目标也调整为同样的tensors
        intersection = (input * target)                         # 计算输入和目标的元素级别的乘积，这是Dice系数的一部分
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)         # 计算Dice损失
        dice = 1 - dice.sum() / num                             # 计算批次的平均Dice损失
        return 0.5 * bce + dice                                 # BCE给予0.5的权重，dice保留原值

class LovaszHingeLoss(nn.Module):       # 继承自PyTorch 的基类 nn.Module
    def __init__(self):
        super().__init__()              # 同理调用父类

    def forward(self, input, target):   # 定义类的前向传播函数
        input = input.squeeze(1)        # 移除输入张量的第二个维度（索引从0开始）
        target = target.squeeze(1)      # 移除目标张量的第二个维度（索引从0开始）
        loss = lovasz_hinge(input, target, per_image=True)  # 调用 lovasz_hinge 函数计算 Lovasz Hinge 损失

        return loss                     # 返回损失值

