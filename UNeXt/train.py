import argparse
import os
from collections import OrderedDict
from glob import glob
import albumentations as albu
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import Flip
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from archs import UNext
# 前面全是import，没啥说的

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

'''
对于parser.add_argument后面的参数，一般有：
name：参数的名称，在命令行中指定参数时使用。
nargs：参数的个数，指定参数接受多少个值。
type：参数的数据类型，指定参数值应该被解析成哪种类型。
default：参数的默认值，在命令行中未指定该参数时使用。
help：参数的帮助文本，用于在命令行中显示参数的说明信息。
action：参数的动作，指定参数的行为。
'''
# 抽提模型时感觉parse方法并不需要
def parse_args():
    parser = argparse.ArgumentParser()      #使用pytorch中的库argparse，用来对超参数进行设置

    parser.add_argument('--name', default=None,                         #设置训练后产生的模型文件名称
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', #设置epoch默认跑100次
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,     #设置批次数量默认为16
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')    #同理是对模型的一些超参数进行设置
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer     随机梯度下降SGD和Adam
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler  是调整学习率的
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()   # 启用 Batch Normalization(BN) 和 Dropout

    pbar = tqdm(total=len(train_loader))    # 创建一个进度条，其总长度即为训练数据集的样本数
    for input, target, _ in train_loader:   # 对训练集的每一个样本都进行迭代，将其传到GPU中
        input = input.cuda()                #输入和标签都要传
        target = target.cuda()

        # 计算模型的输出
        if config['deep_supervision']:      # 设置:如果config设置为深度监督时的输出
            outputs = model(input)
            loss = 0                        # 定义loss并归零
            for output in outputs:          # 对每一个输出计算损失并进行累加
                loss += criterion(output, target)
            loss /= len(outputs)            # 计算平均损失
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)                   # 如果不满足上述config，同样产生一个输出
            loss = criterion(output, target)        # 使用criterion函数计算损失
            iou,dice = iou_score(output, target)    # 计算IoU分数

        # 计算梯度并进行优化
        optimizer.zero_grad()   # 梯度归零(默认条件下的.backward()其梯度计算会累计而不是替换)
        loss.backward()         # 进行反向传播算法，计算出模型参数的梯度
        optimizer.step()        # 调用优化器的.step来更新参数
        # 总结来说就是完成了模型的梯度归零，计算新的梯度，然后更新模型中的参数
        

        avg_meters['loss'].update(loss.item(), input.size(0))   # 使用当前批次的损失来更新平均损失
        avg_meters['iou'].update(iou, input.size(0))            # 使用当前批次的 IoU 分数更新平均 IoU 分数
        
        # 在有序字典中设置平均损失和平均IoU分数，并作为进度条的后缀显示
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    # 返回一个有序字典，包含训练循环结束后的平均损失和平均 IoU 分数
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

# 验证函数的开始，初始化一个字典来保存验证过程中的平均损失、平均 IoU 分数和平均 Dice 分数
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # 切换到测试集的测试模式
    model.eval()    # 不启用 Batch Normalization 和 Dropout

    with torch.no_grad():   # 不计算梯度从而减少内存使用并加速计算
        pbar = tqdm(total=len(val_loader))  # 创建总长度为验证集数据样本数目的进度条
        for input, target, _ in val_loader: #同理传输到GPU
            input = input.cuda()
            target = target.cuda()

            # 计算输出值
            if config['deep_supervision']:  # 和train中的几乎一致
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():                     # 主程序的入口
    config = vars(parse_args()) # 解析命令行参数并转换为字典形式

    if config['name'] is None:  # 同样是判断是否启用名为深度监督
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    # 在`models`目录下创建以模型名称命名的文件夹，如果该文件夹已存在，则不会重新创建

    print('-' * 20) # 打印配制参数
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)    # 将配置参数保存到模型文件夹中的 config.yml 文件中

    # 定义 loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True  # 若有多个GPU可用，则启动cudnn自动优化

    # 创建模型:从预定义的模型架构中加载指定的模型，并传入相应的参数
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()    # 将模型放到GPU上

    # 过滤出需要梯度的模型参数
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':   #根据你的命令行来选择optimizer
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError
    
    # 根据配制选择学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 数据加载部分，先获取所有图片的ID
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 将图片ID分为训练集和测试集
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 定义训练集和测试集的数据预处理步骤
    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])
    # 创建训练集和验证集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    # 初始化训练日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    # 初始化 best_IoU和 trigger
    best_iou = 0
    trigger = 0
    
    # 训练循环
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # 在一个epoch中的训练设置
        train_log = train(config, train_loader, model, criterion, optimizer)
        # 在测试集上评估模型
        val_log = validate(config, val_loader, model, criterion)

        # 更新学习率
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        # 打印训练和验证的损失值和 IoU
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        # 更新训练日志
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        # 将日志保存到csv文件中
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)
        # trigger数值加一，我认为似乎是用于判断是否需要停止epoch的训练了
        trigger += 1
        # 如果当前的 IoU 大于最佳 IoU，则保存模型，并重置 early stopping 的触发器
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # 如果trigger大于一个设定值就会自动停止epoch的进行
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()    # 清空GPU的缓存


if __name__ == '__main__':
    main()
# 新手说明：只有当这个文件被直接运行的时候，main() 函数才会被执行。如果这个文件是被其他文件导入的，那么 main() 函数就不会被执行。
# 这种模式可以让你的 Python 文件既可以被其他文件导入并使用其中的函数或类，也可以被直接运行。
