from transformers import Trainer, TrainerCallback
import torch
import torch.nn.functional as F
import torch.nn as nn

def dice_loss_fn(inputs, targets, smooth = 1e-6):
    #pred: logit
    
    inputs = torch.sigmoid(inputs)
        
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
    return 1 - dice 

def focal_loss_fn(logits, labels, gamma=2, alpha=0.25, smooth=1e-6):
    probs = torch.sigmoid(logits)
    
    # binary cross entropy loss를 계산합니다.
    # positive 클래스의 경우 -log(prob), negative 클래스의 경우 -log(1 - prob)를 계산하여 합칩니다.
    loss = -(labels * torch.log(probs + smooth) + (1 - labels) * torch.log(1 - probs + smooth))
    
    focal_prob = torch.where(labels==1, probs, 1-probs)
    focal_prob = torch.pow( (1-focal_prob) , gamma) * alpha
    
    loss = loss * focal_prob
    
    # 배치 전체의 평균 loss를 계산합니다.
    loss = torch.mean(loss)
    
    return loss



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
    
    
class FocalDiceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # 레이블 가져오기
        outputs = model(**inputs)  # 모델 호출
        logits = outputs.logits #[Batch, 1, W, H]
        
        labels = F.interpolate(labels.float().unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False)
        
        #print(logits.shape) #[Batch, 1, W, H]
        #print(labels.shape) #[Batch, 1, W, H]
        
        focal_loss = focal_loss_fn(logits, labels) #labels.float()
        dice_loss = dice_loss_fn(logits, labels)

        loss = focal_loss + dice_loss

        return (loss, outputs) if return_outputs else loss


class DiceLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # 레이블 가져오기
        outputs = model(**inputs)  # 모델 호출
        logits = outputs.logits #[Batch, 1, W, H]
        
        labels = F.interpolate(labels.float().unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False)
        
        #print(logits.shape) #[Batch, 1, W, H]
        #print(labels.shape) #[Batch, 1, W, H]
        
        # Dice Loss 계산
        loss = dice_loss_fn(logits, labels) #labels.float()

        return (loss, outputs) if return_outputs else loss


AsymmetricLoss = AsymmetricLossOptimized()
class AsymmetricLossTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # 레이블 가져오기
        outputs = model(**inputs)  # 모델 호출
        logits = outputs.logits #[Batch, 1, W, H]
        
        labels = F.interpolate(labels.float().unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False)
        
        #print(logits.shape) #[Batch, 1, W, H]
        #print(labels.shape) #[Batch, 1, W, H]
        loss = AsymmetricLoss(logits, labels.squeeze(1))

        return (loss, outputs) if return_outputs else loss


    



# if __name__=="__main__":
#     logit = torch.randn(5,1,64,64)
#     labels = torch.randint(0,2,(5,64,64))
    
#     AsymmetricLoss
    
#     out = loss(logit,labels)
#     print(out)