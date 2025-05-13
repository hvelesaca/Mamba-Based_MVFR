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
# ViewAvgAggregate Mejorado
# =========================
class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=None, agr_type='mean', use_attention=True):
        super().__init__()
        self.model = model
        # CAMBIO: lifting_net real
        if lifting_net is None:
            self.lifting_net = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512)
            )
        else:
            self.lifting_net = lifting_net
        self.agr_type = agr_type
        # CAMBIO: atención opcional
        self.use_attention = use_attention
        if use_attention:
            self.attention = SimpleAttention(512)
        else:
            self.attention = None

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = batch_tensor(mvimages, dim=1, squeeze=True)
        features = self.model(batched)
        features = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        # CAMBIO: lifting_net real
        features = self.lifting_net(features)
        # CAMBIO: atención
        if self.use_attention and self.attention is not None:
            pooled_view = self.attention(features)
        else:
            if self.agr_type == 'mean':
                pooled_view = torch.mean(features, dim=1)
            elif self.agr_type == 'max':
                pooled_view = torch.max(features, dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation type: {self.agr_type}")
        # CAMBIO: BatchNorm extra después de agregación
        pooled_view = nn.BatchNorm1d(512).to(pooled_view.device)(pooled_view)
        return pooled_view, features

# =========================
# ViewMambaAggregate Mejorado
# =========================
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
        # CAMBIO: lifting_net real
        self.lifting_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.BatchNorm1d(d_model)
        )
        # CAMBIO: atención opcional
        self.use_attention = use_attention
        if use_attention:
            self.attention = SimpleAttention(d_model)
        else:
            self.attention = None

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = batch_tensor(mvimages, dim=1, squeeze=True)
        features = self.model(batched)
        features = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        mamba_out = self.mamba(features)
        # CAMBIO: lifting_net real
        mamba_out = self.lifting_net(mamba_out)
        # CAMBIO: atención
        if self.use_attention and self.attention is not None:
            pooled_view = self.attention(mamba_out)
        else:
            pooled_view = mamba_out[:, -1, :]
        # CAMBIO: BatchNorm extra después de agregación
        pooled_view = nn.BatchNorm1d(mamba_out.shape[-1]).to(pooled_view.device)(pooled_view)
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
        x = x.reshape(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, num_views, C, T, 224, 224)
        pooled_view, features = self.aggregation_model(x)
        inter = self.inter(pooled_view)
        foul_logits = self.foul_branch(inter)
        action_logits = self.action_branch(inter)
        return foul_logits, action_logits

class MultiTaskModelMamba(nn.Module):
    def __init__(self, dropout=0.5):
        super(MultiTaskModelMamba, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        # CAMBIO: lifting_net y atención
        self.aggregation_model = ViewMambaAggregate(
            model=self.backbone,
            d_model=self.feat_dim,
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
        x = x.reshape(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, num_views, C, T, 224, 224)
        pooled_view, features = self.aggregation_model(x)
        inter = self.inter(pooled_view)
        foul_logits = self.foul_branch(inter)
        action_logits = self.action_branch(inter)
        return foul_logits, action_logits
