from utils import *

def instance_to_foreground(outputs, threshold=0.5):
    cls = outputs["class_logits"]          
    masks = torch.sigmoid(outputs["mask_logits"])  

    gate = (cls >= threshold).float().unsqueeze(-1).unsqueeze(-1)
    gated_masks = masks * gate
    final_mask = gated_masks.max(dim=1).values
    return final_mask
