import numpy as np
import torch
from torch import nn
import evaluate
import random
import rasterio
from PIL import Image

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


metric = evaluate.load("mean_iou")
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
            
        print()
        print("computing...")
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=2,
            ignore_index=255,
            reduce_labels=True,
        )
        
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics
