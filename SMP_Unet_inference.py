# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:37:39 2024

@author: user
"""
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

# inference資料與weight.pth路徑
inference_patch_path = r'/work/u5914116/ki67/patch512/d_data'
color_seg_weights = r'/home/u5914116/Harden/chsj/SMP_Unet/unet_color_seg.pth'
boundary_seg_weights = r'/home/u5914116/Harden/chsj/SMP_Unet/unet_boundary_seg.pth'

# 輸出結果
color_seg_output = r'/work/u5914116/ki67/ex12/d_data_inference/color_seg'
boundary_seg_output = r'/work/u5914116/ki67/ex12/d_data_inference/boundary_seg'

# 定義資料集
class CustomDataset(Dataset):
    def __init__(self, images_dir, transform1=None):
        self.images_dir = images_dir
        self.transform1 = transform1
        self.ids = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_file = os.path.join(self.images_dir, self.ids[idx])
        image_src = Image.open(img_file).convert("RGB")
        
        if self.transform1:
            image_norm = self.transform1(image_src)
        return image_norm, img_name

# 轉換
transform1 = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 512x512 pixels
    transforms.ToTensor()
    ])


# DataLoader
inference_dataset = CustomDataset(inference_patch_path,transform1=transform1)
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

def get_class(src,class_type=0):
    output = np.where(src == class_type , 255, 0)
    output = np.uint8(output)
    return output

def join_contours(src,contour1,contour2,width=1):
    output = cv2.drawContours(src, contour1, -1, (0,0,255), width)
    output = cv2.drawContours(output, contour2, -1, (255,0,0), width)
    return output

def predict(model, loader, save_dir="prediction"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, package in enumerate(loader):
            image_norm, img_name  = package[0], package[1]
            #image_norm = image_norm.cuda()
            outputs = model(image_norm)
            masks = torch.argmax(outputs, dim=1)  # Convert probabilities to predictions
            image_norm, masks = (image_norm.squeeze(0)).permute(1,2,0), masks.squeeze(0)
            #image_norm, masks = image_norm.cpu(), masks.cpu()
            image_norm, masks = image_norm.numpy(), masks.numpy()
            seg_class_1, seg_class_2 = get_class(masks,class_type=1), get_class(masks,class_type=2)
            contour_1, _ = cv2.findContours(seg_class_1,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_2, _ = cv2.findContours(seg_class_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            images = np.uint8(image_norm * 255)
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            outputs = join_contours(images,contour_1,contour_2,width=1)
            cv2.imwrite(save_dir+"/" + '{}'.format(img_name[0]),outputs)
            print('Generate predict : {}'.format(img_name[0]))

# 建立模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
color_seg_model = smp.Unet(encoder_name="resnet50",
                           decoder_channels=(512, 256, 64, 32, 16),
                           in_channels=3,
                           classes=3)

boundary_seg_model = smp.Unet(encoder_name="resnet50",
                              decoder_channels=(512, 256, 64, 32, 16),
                              in_channels=3,
                              classes=3)

color_seg_model.to(device)
boundary_seg_model.to(device)
color_seg_model.load_state_dict(torch.load(color_seg_weights,map_location=torch.device('cpu')))
boundary_seg_model.load_state_dict(torch.load(boundary_seg_weights,map_location=torch.device('cpu')))

# 主程式（示例）
if __name__ == "__main__":    
    # 進行預測
    predict(color_seg_model, inference_loader, save_dir=color_seg_output)
    predict(boundary_seg_model, inference_loader, save_dir=boundary_seg_output)
    print("Finsh All Inference")