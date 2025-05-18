import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from mamba_ssm import Mamba
from timm.models.layers import DropPath


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, T, D]
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.norm(attn_output + x)  # Residual connection
        return attn_output.mean(dim=1)  # Pooling temporal ponderado
        
# =========================
# Multi-Head Attention Layer
# =========================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)

# =========================
# LiftingNet Mejorado con LayerNorm
# =========================
class LiftingNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.relu = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

# =========================
# ViewMambaAggregate Mejorado
# =========================
class ViewMambaAggregate2(nn.Module):
    def __init__(self, model, d_model=512, d_state=16, d_conv=4, expand=2, use_attention=True):
        super().__init__()
        self.model = model
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.lifting_net = LiftingNet(d_model)
        self.use_attention = use_attention
        self.attention = MultiHeadAttention(d_model) if use_attention else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = mvimages.view(B * V, C, T, H, W)
        features = self.model(batched)
        features = features.view(B, V, -1)
        mamba_out = self.mamba(features)
        mamba_out = self.lifting_net(mamba_out)
        pooled_view = self.attention(mamba_out) if self.use_attention else mamba_out.mean(dim=1)
        pooled_view = self.norm(pooled_view)
        return pooled_view, mamba_out

class ViewMambaAggregate(nn.Module):
    def __init__(self, model, d_model=512, d_state=16, d_conv=4, expand=2, use_attention=True):
        super().__init__()
        self.model = model
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.lifting_net = LiftingNet(d_model)
        self.use_attention = use_attention
        self.view_attention = MultiHeadAttention(d_model) if use_attention else None
        self.temporal_attention = TemporalAttention(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape
        batched = mvimages.view(B * V, C, T, H, W)
        features = self.model(batched)  # [B*V, D]
        features = features.view(B, V, -1)  # [B, V, D]

        mamba_out = self.mamba(features)  # [B, V, D]
        mamba_out = self.lifting_net(mamba_out)  # [B, V, D]

        # Atención temporal sobre vistas (asumiendo que cada vista representa un segmento temporal)
        temporal_out = self.temporal_attention(mamba_out)  # [B, D]

        # Atención entre vistas
        if self.use_attention:
            view_out = self.view_attention(mamba_out)  # [B, D]
        else:
            view_out = mamba_out.mean(dim=1)  # [B, D]

        # Combinación de atención temporal y atención entre vistas
        pooled_view = self.norm(temporal_out + view_out)  # [B, D]

        return pooled_view, mamba_out
        
# =========================
# Modelo Multi-tarea Mejorado
# =========================
class MultiTaskModelMamba(nn.Module):
    def __init__(self, dropout=0.5, drop_path_rate=0.1):
        super().__init__()
        self.backbone = video_models.mvit_v2_s(weights=video_models.MViT_V2_S_Weights.KINETICS400_V1)
        self.unfreeze_partial_backbone(layers_to_unfreeze=4)

        in_features = self.backbone.head[1].in_features
        self.backbone.head[1] = nn.Linear(in_features, 512)
        self.feat_dim = 512

        self.aggregation_model = ViewMambaAggregate(
            model=self.backbone,
            d_model=self.feat_dim,
            use_attention=True
        )

        self.shared_inter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            DropPath(drop_path_rate)
        )

        self.foul_attention = MultiHeadAttention(self.feat_dim)
        self.action_attention = MultiHeadAttention(self.feat_dim)

        self.foul_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 4)
        )

        self.action_branch = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 8)
        )

    def unfreeze_partial_backbone(self, layers_to_unfreeze=2):
        layers = list(self.backbone.children())[-layers_to_unfreeze:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        batch_size, num_views, C, T, H, W = x.shape
        x = x.view(batch_size * num_views * T, C, H, W)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.view(batch_size, num_views, C, T, 224, 224)

        pooled_view, features = self.aggregation_model(x)
        shared_features = self.shared_inter(pooled_view)

        foul_features = self.foul_attention(features)
        action_features = self.action_attention(features)

        foul_logits = self.foul_branch(shared_features + foul_features)
        action_logits = self.action_branch(shared_features + action_features)

        return foul_logits, action_logits

# =========================
# Ejemplo de Entrenamiento
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModelMamba().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion_foul = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()

    # Ejemplo de datos sintéticos (reemplaza con tu DataLoader real)
    dummy_input = torch.randn(2, 3, 3, 16, 224, 224).to(device)  # [B, V, C, T, H, W]
    dummy_foul_labels = torch.randint(0, 4, (2,)).to(device)
    dummy_action_labels = torch.randint(0, 8, (2,)).to(device)

    model.train()
    optimizer.zero_grad()

    foul_logits, action_logits = model(dummy_input)

    loss_foul = criterion_foul(foul_logits, dummy_foul_labels)
    loss_action = criterion_action(action_logits, dummy_action_labels)
    loss = loss_foul + loss_action

    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Loss total: {loss.item():.4f}, Loss foul: {loss_foul.item():.4f}, Loss action: {loss_action.item():.4f}")
