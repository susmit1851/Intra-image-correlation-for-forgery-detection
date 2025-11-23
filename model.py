from utils import *



class ConvCorrelationBlock(nn.Module):


    def __init__(self, in_channels, hidden=256):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        corr = torch.cat([c3, c5, c7], dim=1)
        corr = self.refine(corr)
        return corr



class CmfdInstanceModel(nn.Module):
    
    def __init__(self, backbone_name="nvidia/mit-b2",
                 num_queries=5, d_model=256):
        super().__init__()
        config = SegformerConfig.from_pretrained(backbone_name)
        config.output_hidden_states = True
        self.encoder = SegformerModel.from_pretrained(backbone_name, config=config)
        self.proj = nn.Conv2d(config.hidden_sizes[-2], d_model, kernel_size=1)
        self.corr_block = ConvCorrelationBlock(d_model, hidden=d_model)
        self.class_head = nn.Sequential(
            nn.Conv2d(d_model, num_queries, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, num_queries, kernel_size=1)
        )

    def forward(self, x):
        B = x.size(0)

        enc_out = self.encoder(x)
        feats = enc_out.hidden_states[-2]  
        feats = self.proj(feats)           

        corr_feats = self.corr_block(feats)

        masks = self.mask_head(corr_feats)
        masks = F.interpolate(masks, size=x.shape[2:], mode="bilinear", align_corners=False)

        class_map = self.class_head(corr_feats)
        class_logits = class_map.view(class_map.size(0), -1)  # (B, num_queries)

        return {
            "mask_logits": masks,
            "class_logits": class_logits
        }




