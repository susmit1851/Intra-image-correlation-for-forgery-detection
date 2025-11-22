from utils import *

class CmfdInstanceModel(nn.Module):
    def __init__(self, backbone_name="nvidia/mit-b2",
                 num_queries=5, d_model=256, nheads=8, num_decoder_layers=2):
        super().__init__()

        config = SegformerConfig.from_pretrained(backbone_name)
        config.output_hidden_states = True
        self.encoder = SegformerModel.from_pretrained(backbone_name, config=config)

        self.proj = nn.Conv2d(config.hidden_sizes[-2], d_model, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nheads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_head = nn.Linear(d_model, 1)
        self.mask_head = nn.Conv2d(d_model, num_queries, kernel_size=1)

    def forward(self, x):
        B = x.size(0)

        enc_out = self.encoder(x)
        assert enc_out.hidden_states is not None, "Hidden states missing — config not set!"

        feats = enc_out.hidden_states[-2]  

        feats = self.proj(feats) 
        B, C, H, W = feats.shape

        src = feats.flatten(2).permute(2, 0, 1)  
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        hs = self.decoder(queries, src)  

        class_logits = self.class_head(hs).sigmoid().permute(1, 0, 2)  
        masks = self.mask_head(feats)  
        masks = F.interpolate(masks, size=x.shape[2:], mode="bilinear", align_corners=False)
        return {
            "class_logits": class_logits.squeeze(-1),
            "mask_logits": masks
        }



