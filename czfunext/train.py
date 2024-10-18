import argparse
import albumentations as albu
import pandas as pd
import torch
import torch.optim as optim
import os
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from method import Unext, BCEDiceLoss, AverageMeter, Dataset
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define for computational work, no need to care about this module
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    # dice = (2* iou) / (iou+1)
    return iou

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    model.train()
    
    pbar = tqdm(total=len(train_loader))
    
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)                   
        loss = criterion(output, target)        # 使用criterion函数计算损失
        iou = iou_score(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        
        pbar.set_postfix(postfix)   # visually work
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
    
def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    model.eval()

    with torch.no_grad(): 
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)                   # 如果不满足上述config，同样产生一个输出
            loss = criterion(output, target)        # 使用criterion函数计算损失
            iou = iou_score(output, target)
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            

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
    
def main():
    parser = argparse.ArgumentParser(description='UNext Training Script')
    parser.add_argument('--data_dir', default='inputs/busi', type=str, help='dataset directory')
    parser.add_argument('--num_workers', default=3, type=int, help='number of workers for data loading')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training and validation')
    parser.add_argument('--epochs', default=500, type=int, help='number of epochs to train')
    parser.add_argument('--img_ext', default= '.png', type= str, help='file img classification')
    parser.add_argument('--mask_ext', default= '.png', type= str, help='file mask classification')
    parser.add_argument('--lr', default= 0.001, type=float, help='learning rate')
    parser.add_argument('--model_save_path', default='./models', type=str, help='path to save the trained model')
    args = parser.parse_args()
    
    os.makedirs(args.model_save_path, exist_ok=True)
    
    model = Unext(num_classes=1, input_channels=3).cuda()
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = BCEDiceLoss().cuda()
    # cudnn.benchmark = True
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=3)
    
    # 数据加载部分，先获取所有图片的ID
    img_ids = glob(os.path.join(args.data_dir, 'images', '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 将图片ID分为训练集和测试集
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 定义训练集和测试集的数据预处理步骤
    train_transform = albu.Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Resize(512, 512),
        albu.Normalize(),
    ])

    val_transform = albu.Compose([
        albu.Resize(512, 512),
        albu.Normalize(),
    ])

    # 创建训练集和验证集
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(args.data_dir, 'images'), 
        mask_dir=os.path.join(args.data_dir, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=1,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(args.data_dir, 'images'),
        mask_dir=os.path.join(args.data_dir, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=1,                                              # 目前做的就是二分类
        transform=val_transform)

    # 创建数据加载器
    # Replace the batch size and num_workers with the parsed arguments
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # use argparse value
        shuffle=True,
        num_workers=args.num_workers,  # use argparse value
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # use argparse value
        shuffle=False,
        num_workers=args.num_workers,  # use argparse value
        drop_last=False)
    
    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    best_iou = 0
    trigger = 0


# 开始训练的循环
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, 500))

    # 在一个epoch中的训练设置
        train_log = train(train_loader, model, criterion, optimizer)
    # 在测试集上评估模型
        val_log = validate(val_loader, model, criterion)

    # 更新学习率
        scheduler.step()
    
    # 打印训练和验证的损失值和 IoU
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
          % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        # 更新训练日志
        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        # 将日志保存到csv文件中
        pd.DataFrame(log).to_csv("diary.csv", index=False)
        
        # trigger数值加一，我认为似乎是用于判断是否需要停止epoch的训练了
        trigger += 1
        # 如果当前的 IoU 大于最佳 IoU，则保存模型，并重置 early stopping 的触发器
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_model.pth'))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        


        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()