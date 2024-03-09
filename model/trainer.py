from transformers import Trainer
import torch
import torch.nn.functional as F

def dice_loss(inputs, targets, smooth = 1e-6):
    #pred: logit
    
    inputs = torch.sigmoid(inputs)
        
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
    return 1 - dice 


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # 레이블 가져오기
        outputs = model(**inputs)  # 모델 호출
        logits = outputs.logits
        
        logits = logits.unsqueeze(1)
        labels = F.interpolate(labels.float().unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False)
        
        # Dice Loss 계산
        loss = dice_loss(logits, labels) #labels.float()

        return (loss, outputs) if return_outputs else loss