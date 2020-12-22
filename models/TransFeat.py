import torch
from einops import rearrange, repeat
from torch import nn

class TransFeat(nn.Module):
    def __init__(self, config):
        super().__init__()

        patch_size = config.get("patch_size")
        height, width, channels = config.get("image_shape")
        assert height*width % patch_size == 0

        num_patches = (height*width//num_patches) ** 2
        patch_dim = channels * (patch_size ** 2)

        self.transformer_dim = config.get("transformer_dim")
        self.patch_embed = nn.Linear(patch_dim, transformer_dim)
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, transformer_dim))

        self.transformer = Transformer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       num_encoder_layers = config.get("num_encoder_layers"),
                                       num_decoder_layers  = config.get("num_decoder_layers"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))

        self.descriptor_dim = config.get("desc_dim")
        self.query_embed = nn.Embedding(config.get("keypoints_num"), self.transformer_dim)

        hidden_dim = self.transformer_dim
        self.kp_head = MLP(hidden_dim, hidden_dim, 3, 3) # x,y, score
        self.desc_head = MLP(hidden_dim, hidden_dim, self.descriptor_dim, 3)


    def forward(self, data):

        p = self.patch_size
        imgs_shapes = data.get('img_shape')

        # split to patches and flatten - code taken from:
        x = rearrange(data.get('img'), 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_embed(x)

        x += self.position_embed[:, :(n + 1)]
        x = self.transformer(x)

        kps = self.kps_head(x).sigmoid()
        scores = kps[:, 2]
        kps = kps[:, :2]
        # From relative [0, 1] to absolute [0, height] coordinates
        h, w = imgs_shapes.unbind(1)
        scale_factor = torch.stack([w, h], dim=1)
        kps = kps * scale_factor[:, None, :]

        descs = self.desc_head(x)
        return kps, scores, descs

    class MLP(nn.Module):
        """ A simple multi-layer perceptron (also called FFN)
            code from:
        """

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x
