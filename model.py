import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models import tiny_vit, convnextv2
from models.tiny_vit import Conv2d_BN
from models import efficientformer_v2

def freeze_pretrained_layers(model):
    """Freeze all layers except the last layer(fc or classifier)"""
    for param in model.parameters():
        param.requires_grad = False
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True

class ConvNext(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        layers = list(base.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(768, 3, bias=True)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)



class Swin(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        base = models.swin_s(weights='DEFAULT')
        layers = list(base.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(768, 3, bias=True)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x)
        return self.classifier(representations)



class BeitV2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.base = timm.create_model('beitv2_base_patch16_224', pretrained=True)
        self.base.head = nn.Linear(768, 3, bias=True)
        freeze_pretrained_layers(self.base)

    def forward(self, x):
        return self.base(x)
    



class Tiny_Vit_21M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit.tiny_vit_21m_224(pretrained=True)
        self.base.head = nn.Linear(576, num_classes, bias=True)
    
    def forward(self, x):
        return self.base(x)

    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)


class Tiny_Vit_11M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit.tiny_vit_11m_224(pretrained=True)
        self.base.head = nn.Linear(448, num_classes, bias=True)
    
    def forward(self, x):
        return self.base(x)

    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)


class Tiny_Vit_5M(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = tiny_vit.tiny_vit_5m_224(pretrained=True)
        self.base.head = nn.Linear(320, num_classes, bias=True)
    
    def forward(self, x):
        return self.base(x)
    
    def fuse_model(self):
        for m in self.base.modules():
            if isinstance(m, Conv2d_BN):
                torch.ao.quantization.fuse_modules(m, ['c', 'bn'], inplace=True)

class ConvNextV2_femto(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = convnextv2.convnextv2_femto()
        if pretrained==True:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt",
                map_location='cpu', check_hash=False,
            )
            self.base.load_state_dict(checkpoint['model'])
        self.base.head = nn.Linear(384, num_classes, bias=True)
    
    def forward(self, x):
        return self.base(x)



class EfficientFormerV2(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        self.base = efficientformer_v2.efficientformerv2_s1(pretrained)
        if pretrained==True:
            checkpoint = torch.load("pretrained/eformer_s1_450.pth", map_location='cpu')
            checkpoint["model"].pop("dist_head.weight")
            checkpoint["model"].pop("dist_head.bias")
            self.base.load_state_dict(checkpoint['model'])
        self.base.head = nn.Linear(224, num_classes, bias=True)

    def forward(self, x):
        return self.base(x)