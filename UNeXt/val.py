# 这就是GPT4.0的代码备注能力
import argparse  # 导入命令行解析模块
import os  # 导入操作系统接口模块
from glob import glob  # 导入文件名模式匹配模块

import cv2  # 导入OpenCV库
import torch  # 导入PyTorch库
import torch.backends.cudnn as cudnn  # 导入CUDA深度神经网络库
import yaml  # 导入YAML解析库
from albumentations.augmentations import transforms  # 导入albumentations库中的transforms模块
from albumentations.core.composition import Compose  # 导入albumentations库的Compose方法
from sklearn.model_selection import train_test_split  # 导入sklearn库的数据集分割函数
from tqdm import tqdm  # 导入进度条库

import archs  # 导入archs模块（可能包含网络架构定义）
from dataset import Dataset  # 从dataset模块导入Dataset类
from metrics import iou_score  # 从metrics模块导入iou_score函数
from utils import AverageMeter  # 从utils模块导入AverageMeter类
from albumentations import RandomRotate90, Resize  # 从albumentations模块导入RandomRotate90和Resize类
import time  # 导入时间模块
from archs import UNext  # 从archs模块导入UNext类（可能是一个模型架构）

def parse_args():
    parser = argparse.ArgumentParser()  # 创建命令行解析器

    parser.add_argument('--name', default=None,
                        help='model name')  # 添加命令行参数--name用于指定模型名称

    args = parser.parse_args()  # 解析命令行参数

    return args  # 返回解析后的参数


def main():
    args = parse_args()  # 解析命令行参数

    with open('models/%s/config.yml' % args.name, 'r') as f:  # 打开配置文件
        config = yaml.load(f, Loader=yaml.FullLoader)  # 使用YAML加载配置文件

    print('-'*20)  # 打印分隔线
    for key in config.keys():  # 遍历配置字典的键
        print('%s: %s' % (key, str(config[key])))  # 打印配置项
    print('-'*20)  # 打印分隔线

    cudnn.benchmark = True  # 启用cudnn加速

    print("=> creating model %s" % config['arch'])  # 打印正在创建的模型信息
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])  # 通过配置创建模型

    model = model.cuda()  # 将模型转移到CUDA

    # 数据加载代码
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))  # 获取所有图片文件路径
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]  # 提取文件名（不包括扩展名）

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)  # 分割数据集，获取验证集图片ID

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))  # 加载模型权重
    model.eval()  # 将模型设置为评估模式

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),  # 设置验证集图片的变换操作：调整大小
        transforms.Normalize(),  # 归一化
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)  # 创建验证集数据集
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],  # 批量大小
       shuffle=False,  # 是否打乱数据
        num_workers=config['num_workers'],  # 使用的工作进程数量
        drop_last=False)  # 是否丢弃最后一个不完整的批次

    iou_avg_meter = AverageMeter()  # 创建IoU平均计量器
    dice_avg_meter = AverageMeter()  # 创建Dice系数平均计量器
    gput = AverageMeter()  # 创建GPU时间平均计量器（未使用）
    cput = AverageMeter()  # 创建CPU时间平均计量器（未使用）

    count = 0  # 计数器初始化（未使用）
    for c in range(config['num_classes']):  # 根据类别数创建输出文件夹
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():  # 不计算梯度，用于推理/评估
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):  # 遍历数据加载器
            input = input.cuda()  # 将输入数据移动到CUDA
            target = target.cuda()  # 将目标数据移动到CUDA
            model = model.cuda()  # 确保模型在CUDA上
            # 计算输出
            output = model(input)

            iou, dice = iou_score(output, target)  # 计算IoU和Dice系数
            iou_avg_meter.update(iou, input.size(0))  # 更新IoU平均计量器
            dice_avg_meter.update(dice, input.size(0))  # 更新Dice系数平均计量器

            output = torch.sigmoid(output).cpu().numpy()  # 将输出通过sigmoid函数激活并移动到CPU，转换为numpy数组
            output[output>=0.5] = 1  # 将概率大于等于0.5的预测设置为1
            output[output<0.5] = 0  # 将概率小于0.5的预测设置为0

            for i in range(len(output)):  # 遍历输出数据
                for c in range(config['num_classes']):  # 遍历每个类别
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),  # 写入输出图片
                                (output[i, c] * 255).astype('uint8'))  # 转换数据类型为uint8

    print('IoU: %.4f' % iou_avg_meter.avg)  # 打印IoU平均值
    print('Dice: %.4f' % dice_avg_meter.avg)  # 打印Dice系数平均值

    torch.cuda.empty_cache()  # 清空CUDA缓存


if __name__ == '__main__':
    main()  # 如果是主程序，执行main函数

'''
import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
'''