import torch
from vit_pytorch import ViT
import torch
import torch.nn.functional as F

def vit_resize(x):
    # 1. 通道扩展：将 1 通道扩展为 3 通道
    x = x.repeat(1, 3, 1, 1)  # 现在的形状为 [64, 3, 126, 1200]

    # 2. 调整尺寸：使用 interpolate 函数将 (126, 1200) 调整到 (224, 224)
    x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return x_resized

#
#
# v = ViT(
#     image_size = 224,
#     patch_size = 32,
#     num_classes = 50,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

def Process_ViT(num_classes):
    model = ViT(
        image_size=224,
        patch_size=32,
        num_classes=num_classes,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return model

if __name__ == '__main__':
    x = torch.randn(64, 1, 126, 400)  # 生成随机 EEG 数据
    x = vit_resize(x)
    print(x.shape)