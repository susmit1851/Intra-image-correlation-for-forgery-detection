from utils import *

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        
        dice_per_sample = 1 - (2. * inter + smooth) / (union + smooth)  
        return dice_per_sample.mean()   


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy(prob, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


class CmfdLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = SigmoidFocalLoss()
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, outputs, targets):
        masks = outputs["mask_logits"]   
        cls = outputs["class_logits"]    

        mask_pred = masks.max(dim=1).values.unsqueeze(1)  

        loss_dict = {}
        loss_dict["dice"] = self.dice(mask_pred, targets)   
        loss_dict["focal"] = self.focal(mask_pred, targets) 

        loss = self.dw * loss_dict["dice"] + self.fw * loss_dict["focal"]
        return loss, loss_dict

