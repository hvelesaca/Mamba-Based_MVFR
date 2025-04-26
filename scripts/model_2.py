import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torch.nn.functional as F
from mamba_ssm import Mamba

# Utility functions
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

# Existing ViewAvgAggregate
class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential(), agr_type='mean'):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.agr_type = agr_type

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = batch_tensor(mvimages, dim=1, squeeze=True)
        features = self.model(batched)
        features = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        features = self.lifting_net(features)
        if self.agr_type == 'mean':
            pooled_view = torch.mean(features, dim=1)
        elif self.agr_type == 'max':
            pooled_view = torch.max(features, dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation type: {self.agr_type}")
        return pooled_view, features

# Updated ViewMambaAggregate
class ViewMambaAggregate(nn.Module):
    def __init__(self, model, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.model = model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.lifting_net = nn.Sequential()

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = batch_tensor(mvimages, dim=1, squeeze=True)  # [B*V*T, C, H, W]
        features = self.model(batched)  # [B*V, feat_dim]
        features = unbatch_tensor(features, B, dim=1, unsqueeze=True)  # [B, V, feat_dim]
        
        # Ensure features are in [B, L, D] format for Mamba
        mamba_out = self.mamba(features)  # [B, V, feat_dim]
        pooled_view = mamba_out[:, -1, :]  # [B, feat_dim]
        features = self.lifting_net(mamba_out)
        return pooled_view, features

# Existing models (unchanged)
class SimpleFoulModel(nn.Module):
    def __init__(self, agr_type='mean'):
        super(SimpleFoulModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        self.aggregation_model = ViewAvgAggregate(model=self.backbone, lifting_net=nn.Sequential(), agr_type=agr_type)
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
            nn.Dropout(0.3),
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
    def __init__(self, agr_type='mean'):
        super(SimpleActionModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        self.aggregation_model = ViewAvgAggregate(model=self.backbone, lifting_net=nn.Sequential(), agr_type=agr_type)
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
            nn.Dropout(0.3),
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
    def __init__(self, agr_type='mean'):
        super(MultiTaskModel, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        self.aggregation_model = ViewAvgAggregate(model=self.backbone, lifting_net=nn.Sequential(), agr_type=agr_type)
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
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        self.action_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
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

# Updated MultiTaskModelMamba
class MultiTaskModelMamba(nn.Module):
    def __init__(self):
        super(MultiTaskModelMamba, self).__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512
        self.aggregation_model = ViewMambaAggregate(model=self.backbone, d_model=self.feat_dim)
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
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
        self.action_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
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
    
