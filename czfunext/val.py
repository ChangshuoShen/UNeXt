import os
from glob import glob
import cv2
import torch
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from albumentations import Resize
from method import Unext, Dataset, AverageMeter

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
    dice = (2* iou) / (iou+1)
    return iou


def main():
    
    model = Unext(num_classes=1, input_channels=3).cuda()
    
    img_ids = glob(os.path.join('inputs', 'val_dataset', 'images', '*' + '.jpg'))    ########### 修改数据库读取路径和文件格式
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     'my_val'))
    model.eval()
    
    val_transform = Compose([           # 这个地方为什么要resize ?
        Resize(512, 512),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'val_dataset', 'images'),        ##### 在此处修改文件路径
        mask_dir=os.path.join('inputs', 'val_dataset', 'masks'),
        img_ext='.jpg',         ###### 在这里修改图片后缀读取
        mask_ext='.png',
        num_classes=1,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = 16,
        shuffle=False,
        num_workers= 3,
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(1):  ######### 输入的是num_class
        os.makedirs(os.path.join('outputs', 'my_val', str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)


            iou = iou_score(output, target)
            dice = (2* iou) / (iou+1)       ###### 手动修改的
            
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(1):
                    cv2.imwrite(os.path.join('outputs', 'my_valresult', str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()