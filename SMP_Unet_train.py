# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:00:42 2024

@author: user
"""
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm

# 訓練資料路徑
label_path = r'/work/u5914116/ki67/ex14/label'
patch_path = r'/work/u5914116/ki67/ex14/patch'

# Pretrained weight
encoder_weights = r'/home/u5914116/Harden/chsj/SMP_Unet/resnet50_encoder_2.pth'
color_seg_weights = r'/home/u5914116/Harden/chsj/SMP_Unet/best_color_weights_ex1.pth'

# 定義資料集
class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform1=None, transform2=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.ids = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_file = os.path.join(self.images_dir, self.ids[idx])
        mask_file = os.path.join(self.masks_dir, self.ids[idx])
        image = Image.open(img_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")
        
        if self.transform1:
            image = self.transform1(image)
            mask_color = (self.transform2(mask) * 255).to(torch.uint8)
            mask_tmp = (mask_color.clone()).numpy()
            mask_border = get_mask_bor(mask_tmp)
            mask_border = (torch.from_numpy(mask_border)).to(torch.uint8)
        return image, mask_color, mask_border

# 轉換
transform1 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ColorJitter(brightness=(0.3,1),
                           hue=(-0.1,0)),
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    ])

def get_bor(mask):
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    dilate_mask = cv2.dilate(mask, cross_kernel,iterations = 2)
    erode_mask = cv2.erode(mask, cross_kernel,iterations = 2)
    dilate_mask = np.squeeze(dilate_mask)
    erode_mask = np.squeeze(erode_mask)
    border = dilate_mask - erode_mask
    if len(border.shape) < 3:
        border = border[np.newaxis,:,:]
    border[border==255] = 2
    return border

def get_seg(mask,class_type=0):
    output = np.where(mask==class_type,255,0)
    output = np.uint8(output)
    return output

def get_cell(mask):
    output = np.where(mask==0,0,1)
    output = np.uint8(output)
    return output

def get_mask_bor(mask):
    mask = np.squeeze(mask)
    red_seg, blue_seg = get_seg(mask,class_type=1), get_seg(mask,class_type=2)
    red_bor, blue_bor = get_bor(red_seg), get_bor(blue_seg)
    
    mix_bor = red_bor + blue_bor
    mix_bor[mix_bor>2] = 2
    
    mix_cell = get_cell(mask)
    output = mix_bor + mix_cell
    output[output>2] = 2
    return output

# DataLoader
dataset = CustomDataset(patch_path, label_path, transform1=transform1,transform2=transform2)
train_num = int(len(dataset) * 0.8)
train_dataset, other_dataset = random_split(dataset,[train_num, len(dataset) - train_num])
val_num = int(len(other_dataset) * 0.5)
val_dataset, test_dataset = random_split(other_dataset,[val_num, len(other_dataset) - val_num])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 建立模型

# Color segmentation model
model_color = smp.Unet(encoder_name="resnet50",
                       decoder_channels=(512, 256, 64, 32, 16),
                       in_channels=3,
                       classes=3)

for param in model_color.encoder.parameters():
    param.requires_grad = False

model_color.cuda()
model_color.encoder.load_state_dict(torch.load(encoder_weights))

# Border segmentation model
model_border = smp.Unet(encoder_name="resnet50",
                        decoder_channels=(512, 256, 64, 32, 16),
                        in_channels=3,
                        classes=3)

for param in model_border.encoder.parameters():
    param.requires_grad = False
    
model_border.cuda()
model_border.load_state_dict(torch.load(color_seg_weights))

# 損失函數和優化器
def Weight_Dice_loss(predicted, target, num_classes=3, smooth=1e-6):
    smooth = 1e-6
    losses = []

    for class_index in range(num_classes):
        predicted_class = predicted[:, class_index]  # size = (batch,512,512) 
        target_class = (target == class_index).float()  # size = (batch,1,512,512)
        #batch_size = predicted_class.size(0)

        predicted_class = predicted_class.view(predicted_class.size(0), -1) # flatten(512x512 => 262144)
        target_class = target_class.view(target_class.size(0), -1) # flatten(512x512 => 262144)

        intersection = torch.sum(predicted_class * target_class) # 以batch為單位計算交集
        union = torch.sum(predicted_class) + torch.sum(target_class) # 以batch為單位計算集
        dice = (2.0 * intersection) / (union + smooth)
        dice_loss = 1.0 - dice
        losses.append(dice_loss)

    loss = sum(losses) / num_classes

    return loss

def Weight_CE_loss(predicted, target, num_classes=3, smooth=1e-6):
    smooth = 1e-6
    gamma = 2
    losses = []

    for class_index in range(num_classes):
        predicted_class = predicted[:, class_index]
        target_class = (target == class_index).float()
        batch_size = predicted_class.size(0)

        predicted_class = predicted_class.view(predicted_class.size(0), -1) # flatten(512x512 => 262144)
        target_class = target_class.view(target_class.size(0), -1) # flatten(512x512 => 262144)

        pred_log = torch.log(predicted_class + smooth) # 以batch為單位計算log
        comp_pred_log = torch.log(1 - predicted_class + smooth) # 以batch為單位計算log
        scale_1 = (1 - predicted_class) ** gamma
        scale_2 = predicted_class ** gamma
        term_1  = torch.sum(scale_1 * target_class * pred_log) / (262144 * batch_size)
        term_2 = torch.sum(scale_2 * (1 - target_class) * comp_pred_log) / (262144 * batch_size)
        loss = term_1 + term_2
        losses.append(loss)

    loss = - (sum(losses) / num_classes)

    return loss

criterion_1, criterion_2 = Weight_Dice_loss, Weight_CE_loss
optimizer_color = torch.optim.Adam(model_color.parameters(), lr=1e-4)
optimizer_border = torch.optim.Adam(model_border.parameters(), lr=1e-4)

# 訓練函式
def train_model(model_1, model_2, optimizer_1, optimizer_2, train_loader, val_loader, num_epochs=25, n_classes=3):
    best_miou_1, best_miou_2 = 0.0, 0.0
    best_weights_1, best_weights_2 = None, None

    for epoch in range(num_epochs):
        model_1.train()
        model_2.train()
        running_loss_1, running_loss_2 = 0.0, 0.0
        count = 0
        tbar = tqdm(train_loader)
        # Training loop
        for images, mask_color, mask_border in tbar:
            images = images.cuda()
            mask_color = mask_color.cuda()
            mask_border = mask_border.cuda()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            
            # model 1
            outputs_1 = model_1(images)
            pred_1 = torch.softmax(outputs_1, dim=1)
            loss_1 = criterion_1(pred_1, mask_color) + criterion_2(pred_1, mask_color)
            loss_1.backward()
            optimizer_1.step()
                
            # model 2
            outputs_2 = model_2(images)
            pred_2 = torch.softmax(outputs_2, dim=1)
            loss_2 = criterion_1(pred_2, mask_border) + criterion_2(pred_2, mask_border)
            loss_2.backward()
            optimizer_2.step()
            
            running_loss_1 += loss_1.item()
            running_loss_2 += loss_2.item()
            tbar.set_description('Color Loss: %.3f / Border Loss: %.3f' % (loss_1,loss_2))

        avg_loss_1 = running_loss_1 / len(train_loader)
        avg_loss_2 = running_loss_2 / len(train_loader)

        # Evaluation loop
        val_miou_1, val_miou_2 = evaluate_model(model_1, model_2, val_loader, n_classes=n_classes)
        print(f"Epoch {epoch+1}/{num_epochs}, Color / Border Loss : {avg_loss_1:.4f} / {avg_loss_2:.4f}, Val Color / Border MIoU : {val_miou_1:.4f} / {val_miou_2:.4f}")

        # Update the best model if the current model is better
        if (val_miou_1 > best_miou_1):
            best_miou_1 = val_miou_1
            best_weights_1 = model_1.state_dict().copy()
        if (val_miou_2 > best_miou_2):
            best_miou_2 = val_miou_2
            best_weights_2 = model_2.state_dict().copy()

    # After all epochs, load the best model weights
    model_1.load_state_dict(best_weights_1)
    model_2.load_state_dict(best_weights_2)
    # Optionally, save the best model to disk
    torch.save(best_weights_1, "best_color_weights.pth")
    torch.save(best_weights_2, "best_border_weights.pth")
    print(f"Training complete. Best Val Color / Border MIoU: {best_miou_1:.4f} / {best_miou_2:.4f}")

    return model_1, model_2  # Return the model with the best weights

#import torch
from torchvision.utils import save_image

def miou_score(pred, target, smooth=1e-6, n_classes=3):
    """
    Compute the Mean Intersection over Union (MIoU) score.
    :param pred: the model's predicted probabilities
    :param target: the ground truth
    :param smooth: a small value to avoid division by zero
    :param n_classes: the number of classes in the dataset
    :return: the MIoU score
    """
    pred = torch.argmax(pred, dim=1)
    miou_total = 0.0
    for class_id in range(n_classes):
        true_positive = ((pred == class_id) & (target == class_id)).sum()
        false_positive = ((pred == class_id) & (target != class_id)).sum()
        false_negative = ((pred != class_id) & (target == class_id)).sum()
        intersection = true_positive
        union = true_positive + false_positive + false_negative + smooth
        miou = intersection / union
        miou_total += miou
    return miou_total / n_classes

def evaluate_model(model_1, model_2, loader, n_classes=3):
    model_1.eval()
    model_2.eval()
    total_miou_1, total_miou_2 = 0, 0
    tbar = tqdm(loader)
    with torch.no_grad():
        for images, mask_color, mask_border in tbar:
            images = images.cuda()
            mask_color = mask_color.cuda()
            mask_border = mask_border.cuda()
            outputs_1, outputs_2 = model_1(images), model_2(images)
            total_miou_1 += miou_score(outputs_1, mask_color, n_classes=n_classes)
            total_miou_2 += miou_score(outputs_2, mask_border, n_classes=n_classes)
            
    return (total_miou_1 / len(loader))*100, (total_miou_2 / len(loader))*100

def predict(model, loader, save_dir="predicted_masks"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, package in enumerate(loader):
            images = package[0]
            images = images.cuda()
            outputs = model(images)
            masks = torch.argmax(outputs, dim=1)
            
            r_channel = (torch.where(masks == 1,
                               torch.tensor(255,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)
            
            g_channel = (torch.where(masks == 0,
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)
            
            b_channel = (torch.where(masks == 2,
                               torch.tensor(255,dtype=torch.uint8,device = 'cuda'),
                               torch.tensor(0,dtype=torch.uint8,device = 'cuda')
                               )).unsqueeze(1)

            preds = torch.cat((r_channel,g_channel),1)
            preds = torch.cat((preds,b_channel),1)
            
            for j, pred in enumerate(preds):
                save_image(pred.float(), os.path.join(save_dir, f"predict_{idx * loader.batch_size + j}.png"))

# 主程式
if __name__ == "__main__":

    model_color, model_border = train_model(model_color, model_border, optimizer_color, optimizer_border,
                                            train_loader,
                                            val_loader,
                                            num_epochs=30)
    
    test_MIoU_color, test_MIoU_border = evaluate_model(model_color, model_border, test_loader)
    print(f"test Color / Border MIoU: {test_MIoU_color} / {test_MIoU_border}")
    predict(model_color, test_loader, save_dir="color_predicted_masks")
    predict(model_border, test_loader, save_dir="border_predicted_masks")
    print("Finsh All Training and Testing")
