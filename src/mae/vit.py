import timm
import torch


class ViTBlocks(torch.nn.Module):
    '''The main processing blocks of ViT. Excludes things like patch embedding and classificaton
    layer.
    Args:
        width: size of the feature dimension.
        depth: number of blocks in the network.
        end_norm: whether to end with LayerNorm or not.
    '''
    def __init__(
        self,
        width: int = 768,
        depth: int = 12,
        end_norm: bool = True,
    ):
        super().__init__()

        # transformer blocks from ViT
        ViT = timm.models.vision_transformer.VisionTransformer
        vit = ViT(embed_dim=width, depth=depth)
        self.layers = vit.blocks
        if end_norm:
            # final normalization
            self.layers.add_module('norm', vit.norm)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
