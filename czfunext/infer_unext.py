import os.path as osp
import cv2
import torch
from albumentations import Resize, Compose
from albumentations.augmentations import transforms
from method import Unext  # 确保这里包含了你自定义的模型类
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
# 定义全局的device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义推理函数，处理单张图片
def infer_single_image(model_path, input_path, output_path, num_classes=1):
    """
    对单张图片进行推理，并将输出结果保存到指定路径。
    :param model_path: 预训练模型的路径 (.pth 文件)
    :param input_path: 输入图片的路径
    :param output_path: 推理结果保存路径
    :param num_classes: 模型的输出类别数
    """
    # 加载模型
    model = Unext(num_classes=num_classes, input_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # 图像预处理（调整大小并进行归一化）
    transform = Compose([
        Resize(512, 512),  # 根据模型要求调整图像大小
        transforms.Normalize()
    ])

    # 读取输入图片
    image = cv2.imread(input_path)  # BGR 格式
    if image is None:
        raise ValueError(f"无法读取输入图片: {input_path}")

    # 进行图像预处理
    augmented = transform(image=image)
    image = augmented['image']
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # 扩展为 batch 维度

    with torch.no_grad():
        # 推理
        output = model(image)
        output = torch.sigmoid(output).cpu().numpy()

        # 阈值化处理 (二分类情况下，0.5 作为阈值)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        # 保存输出结果
        output_image = (output[0, 0] * 255).astype('uint8')  # [0, 1] -> [0, 255]
        cv2.imwrite(output_path, output_image)

    print(f"推理完成，结果已保存到: {output_path}")

# 定义请求体数据模型
class InferenceRequest(BaseModel):
    input_path: str       # 输入图片路径
    output_path: str      # 输出结果保存路径
    num_classes: int      # 类别数
    
app = FastAPI()

@app.post("/infer")
async def infer(
    request: InferenceRequest
):
    # 从请求体中提取参数
    input_path = request.input_path
    output_path = request.output_path
    num_classes = request.num_classes
    
    # 校验参数
    if not osp.exists(input_path):
        raise HTTPException(status_code=400, detail="输入文件路径无效")
    
    try:
        # 调用推理函数
        model_path = '/home/shenc/Desktop/IGEM/UNeXt/UNeXt/models/exp_BUSI/model.pth'  # 替换为实际模型路径
        infer_single_image(model_path, input_path, output_path, num_classes)
        return {"message": "推理完成", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理过程中发生错误: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.4", port=8000)