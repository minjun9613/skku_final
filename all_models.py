# all_models.py (통합 모델 정의 포함 ablation 버전들)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import math

def get_eca_kernel_size(channels):
    t = int(abs(math.log2(channels) / 2 + 0.5))
    k = t if t % 2 else t + 1
    return k

class ModelConfig:
    def __init__(self, 
                 num_classes=6,
                 use_pretrained=True,
                 dropout_rate=0.5,
                 hidden_dim=512,
                 fusion_heads=4):
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.fusion_heads = fusion_heads

# MobileNetV2
class MobileNetV2Classifier(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.model = models.mobilenet_v2(weights='DEFAULT' if config.use_pretrained else None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, config.num_classes)

    def forward(self, x):
        return self.model(x)

# ResNet-50
class ResNet50Classifier(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT' if config.use_pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)

    def forward(self, x):
        return self.model(x)

# ViT
class ViTClassifier(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        vit_config = ViTConfig()
        self.vit = ViTModel(vit_config)
        self.classifier = nn.Sequential(
            nn.Linear(vit_config.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ✅ 커널 사이즈를 3으로 고정
        k_size = 3
        self.eca_conv = nn.Conv1d(
            in_dim, in_dim,
            kernel_size=k_size,
            padding=k_size // 2,
            groups=in_dim,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, W, H = x.size()

        # Self-Attention: 공간 간 상호작용
        query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, W, H)
        out = self.gamma * out + x  # residual connection

        # ECA: 채널 간 중요도 반영
        y = self.avg_pool(out)                  # [B, C, 1, 1]
        y = y.squeeze(-1).squeeze(-1)           # [B, C]
        y = self.eca_conv(y.unsqueeze(-1))      # [B, C, 1]
        y = self.sigmoid(y)
        out = out * y.unsqueeze(-1)             # [B, C, 1, 1] broadcasted

        return out


# AttentionFusion
class AttentionFusion(nn.Module):
    def __init__(self, resnet_dim, vit_dim, num_heads=4):
        super().__init__()
        self.resnet_to_vit = nn.Linear(resnet_dim, vit_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=vit_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(vit_dim * 2, 512), nn.ReLU(), nn.Linear(512, 2), nn.Softmax(dim=1)
        )
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim * 2)

    def forward(self, resnet_feat, vit_feat):
        resnet_feat = self.norm1(self.resnet_to_vit(resnet_feat)).unsqueeze(1)
        vit_feat = vit_feat.unsqueeze(1)
        fused, _ = self.multihead_attn(resnet_feat, vit_feat, vit_feat)
        combined = torch.cat((fused.squeeze(1), vit_feat.squeeze(1)), dim=1)
        combined = self.norm2(combined)
        weights = self.fc(combined)
        r_w, v_w = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        return r_w * resnet_feat.squeeze(1) + v_w * vit_feat.squeeze(1)

# 하이브리드 계열 모델 정의 (Ablation 포함)
def make_hybrid_model(attention=False, fusion=False, config=ModelConfig()):
    class HybridAblation(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = models.resnet50(weights='DEFAULT' if config.use_pretrained else None)
            layers = list(resnet.children())[:-2]
            if attention:
                layers.append(HybridAttentionLayer(2048))
            self.resnet_layers = nn.Sequential(*layers)

            self.additional_conv = nn.Sequential(
                nn.Conv2d(2048, 1024, 5, padding=2), nn.BatchNorm2d(1024), nn.ReLU(),
                nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 3, 1), nn.BatchNorm2d(3), nn.ReLU()
            )

            self.resize = nn.AdaptiveAvgPool2d((224, 224))
            vit_config = ViTConfig()
            self.vit = ViTModel(vit_config)
            if fusion:
                self.fusion = AttentionFusion(2048 * 7 * 7, vit_config.hidden_size, num_heads=config.fusion_heads)
                self.classifier = nn.Sequential(
                    nn.Linear(vit_config.hidden_size, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(config.hidden_dim, config.num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(vit_config.hidden_size + 2048 * 7 * 7, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(config.hidden_dim, config.num_classes)
                )

        def forward(self, x):
            r_feat = self.resnet_layers(x)
            att_map = self.additional_conv(r_feat)
            r_feat_flat = r_feat.reshape(r_feat.size(0), -1)
            vit_input = self.resize(att_map).view(x.size(0), 3, 224, 224)
            vit_feat = self.vit(pixel_values=vit_input).last_hidden_state[:, 0, :]
            if hasattr(self, 'fusion'):
                fused = self.fusion(r_feat_flat, vit_feat)
                return self.classifier(fused), att_map
            else:
                combined = torch.cat([r_feat_flat, vit_feat], dim=1)
                return self.classifier(combined), att_map

    return HybridAblation()

# 모델 선택 함수
def get_model(name='hybrid', config=None):
    if config is None:
        config = ModelConfig()
        
    if name == 'mobilenet':
        return MobileNetV2Classifier(config)
    elif name == 'resnet':
        return ResNet50Classifier(config)
    elif name == 'vit':
        return ViTClassifier(config)
    elif name == 'hybrid':
        return make_hybrid_model(attention=True, fusion=True, config=config)
    elif name == 'hybrid_attn_only':
        return make_hybrid_model(attention=True, fusion=False, config=config)
    elif name == 'hybrid_fusion_only':
        return make_hybrid_model(attention=False, fusion=True, config=config)
    elif name == 'hybrid_plain':
        return make_hybrid_model(attention=False, fusion=False, config=config)
    else:
        raise ValueError(f"Unknown model name: {name}")
