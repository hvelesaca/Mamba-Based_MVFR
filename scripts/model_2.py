import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torch.nn.functional as F
from mamba_ssm import Mamba

# =========================
# Utility functions
# =========================
def batch_tensor(tensor, dim=1, squeeze=False):
    shape = list(tensor.shape)
    B = shape[0]
    V = shape[dim]
    shape[0] = B * V
    if squeeze:
        shape.pop(dim)
    else:
        shape[dim] = 1
    return tensor.reshape(shape)

def unbatch_tensor(tensor, B, dim=1, unsqueeze=False):
    shape = list(tensor.shape)
    V = shape[0] // B
    shape[0] = B
    if unsqueeze:
        shape.insert(dim, V)
    else:
        shape[0] = B * V
    return tensor.reshape(shape)

# =========================
# Simple Attention Layer (NUEVO)
# =========================
class SimpleAttention(nn.Module):
    """
    Simple self-attention layer for sequence aggregation.
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [B, V, D]
        Q = self.query(x)
        K = self.key(x)
        Vv = self.value(x)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)
        attn_weights = self.softmax(attn_scores)
        out = torch.bmm(attn_weights, Vv)
        # Aggregate over views (mean)
        return out.mean(dim=1)

# =========================
# LiftingNet: lifting_net con BatchNorm1d correcto
# =========================
class LiftingNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(d_model)
    def forward(self, x):
        # x: [B, V, D]
        B, V, D = x.shape
        x = self.linear(x)
        x = self.relu(x)
        x = x.view(B * V, D)
        x = self.bn(x)
        x = x.view(B, V, D)
        return x
        
class ViewAvgAggregate(nn.Module):
    def __init__(self, model, d_model=512, agr_type='mean', use_attention=True):
        super().__init__()
        self.model = model
        self.lifting_net = LiftingNet(d_model)
        self.agr_type = agr_type
        self.use_attention = use_attention
        if use_attention:
            self.attention = SimpleAttention(d_model)
        else:
            self.attention = None
        self.bn_pool = nn.BatchNorm1d(d_model)

    def forward(self, mvimages):
        # mvimages: [B, V, C, T, H, W]
        B, V, C, T, H, W = mvimages.shape
        # CAMBIO: Solo fusiona batch y views, NO el tiempo
        batched = mvimages.view(B * V, C, T, H, W)  # [B*V, C, T, H, W]
        features = self.model(batched)               # [B*V, feat_dim]
        features = features.view(B, V, -1)           # [B, V, feat_dim]
        features = self.lifting_net(features)
        if self.use_attention and self.attention is not None:
            pooled_view = self.attention(features)
        else:
            if self.agr_type == 'mean':
                pooled_view = torch.mean(features, dim=1)
            elif self.agr_type == 'max':
                pooled_view = torch.max(features, dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation type: {self.agr_type}")
        pooled_view = self.bn_pool(pooled_view)
        return pooled_view, features

class ViewMambaAggregate(nn.Module):
    def __init__(self, model, d_model=512, d_state=16, d_conv=4, expand=2, use_attention=True):
        super().__init__()
        self.model = model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.lifting_net = LiftingNet(d_model)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SimpleAttention(d_model)
        else:
            self.attention = None
        self.bn_pool = nn.BatchNorm1d(d_model)

    def forward(self, mvimages):
        # mvimages: [B, V, C, T, H, W]
        B, V, C, T, H, W = mvimages.shape
        # CAMBIO: Solo fusiona batch y views, NO el tiempo
        batched = mvimages.view(B * V, C, T, H, W)  # [B*V, C, T, H, W]
        features = self.model(batched)               # [B*V, feat_dim]
        features = features.view(B, V, -1)           # [B, V, feat_dim]
        mamba_out = self.mamba(features)
        mamba_out = self.lifting_net(mamba_out)
        if self.use_attention and self.attention is not None:
            pooled_view = self.attention(mamba_out)
        else:
            pooled_view = mamba_out[:, -1, :]
        pooled_view = self.bn_pool(pooled_view)
        return pooled_view, mamba_out
        
# =========================
# Modelos mejorados
# =========================

class SimpleFoulModel(nn.Module):
    def __init__(self, agr_type='mean', dropout=0.4):
        super(SimpleFoulModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        # CAMBIO: lifting_net y atención
        self.aggregation_model = ViewAvgAggregate(
            model=self.backbone,
            lifting_net=None,
            agr_type=agr_type,
            use_attention=True
        )
        self.inter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )
        self.foul_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # CAMBIO: dropout configurable
            nn.Linear(256, 4)
        )

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        batch_size, num_views, C, T, H, W = x.shape
        x = x.reshape(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, num_views, C, T, 224, 224)
        pooled_view, features = self.aggregation_model(x)
        inter = self.inter(pooled_view)
        foul_logits = self.foul_branch(inter)
        return foul_logits

class SimpleActionModel(nn.Module):
    def __init__(self, agr_type='mean', dropout=0.4):
        super(SimpleActionModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        # CAMBIO: lifting_net y atención
        self.aggregation_model = ViewAvgAggregate(
            model=self.backbone,
            lifting_net=None,
            agr_type=agr_type,
            use_attention=True
        )
        self.inter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )
        self.action_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # CAMBIO: dropout configurable
            nn.Linear(256, 8)
        )

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        batch_size, num_views, C, T, H, W = x.shape
        x = x.reshape(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, num_views, C, T, 224, 224)
        pooled_view, features = self.aggregation_model(x)
        inter = self.inter(pooled_view)
        action_logits = self.action_branch(inter)
        return action_logits

class MultiTaskModel(nn.Module):
    def __init__(self, agr_type='mean', dropout=0.4):
        super(MultiTaskModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        # CAMBIO: lifting_net y atención
        self.aggregation_model = ViewAvgAggregate(
            model=self.backbone,
            lifting_net=None,
            agr_type=agr_type,
            use_attention=True
        )
        self.inter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
        )
        self.foul_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # CAMBIO: dropout configurable
            nn.Linear(256, 4)
        )
        self.action_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # CAMBIO: dropout configurable
            nn.Linear(256, 8)
        )

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        batch_size, num_views, C, T, H, W = x.shape
        # CAMBIO: Si necesitas resize espacial, hazlo así:
        x = x.view(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.view(batch_size, num_views, C, T, 224, 224)
        # NO modifiques la dimensión temporal, solo batch y views
        pooled_view, features = self.aggregation_model(x)
        inter = self.inter(pooled_view)
        foul_logits = self.foul_branch(inter)
        action_logits = self.action_branch(inter)
        return foul_logits, action_logits
