import numpy as np
import torch
from torch import nn
import evaluate
import random
import rasterio
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

MAX_PIXEL_VALUE = 65535

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_labels = logits_tensor.squeeze(1).detach().cpu().numpy()
        
        pred_labels = np.where(pred_labels>0, 255, 0)
        labels = np.where(labels>0,255,0)

        for i in range(5):  # 예측된 이미지 중 처음 5개만 저장
            pred_img = Image.fromarray(pred_labels[i].astype(np.uint8))
            label_img = Image.fromarray(labels[i].astype(np.uint8))
            
            pred_img.save(f"example/pred_image_{i}.png")
            label_img.save(f"example/label_image_{i}.png")


        TP = np.sum(np.logical_and(pred_labels == 255, labels == 255))
        FP = np.sum(np.logical_and(pred_labels == 255, labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, labels == 255))

        # IOU 계산
        iou = TP / (TP + FP + FN)

        # Precision 계산
        precision = TP / (TP + FP)

        # Recall 계산
        recall = TP / (TP + FN)

        return  {"IOU": iou,
                 "Precision": precision,
                 "Recall": recall}
