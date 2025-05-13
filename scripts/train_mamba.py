import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_2 import SimpleFoulModel, SimpleActionModel, MultiTaskModel, MultiTaskModelMamba
from tqdm import tqdm
import numpy as np
import kornia.augmentation as K

# CAMBIO: Importar augmentaciones adicionales y focal loss
from torchvision.transforms import RandomErasing
from torch.cuda.amp import autocast, GradScaler

# CAMBIO: Focal Loss (implementación simple)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# CAMBIO: Mixup y Cutmix helpers
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ... (El resto de la clase MVFoulDataset y utilidades no cambia) ...

# CAMBIO: Añadir más augmentations y opción de mixup/cutmix
def get_augmentations(device, use_extra_aug=True):
    aug_list = [
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5)
    ]
    if use_extra_aug:
        aug_list += [
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),  # NUEVO: Ruido gaussiano
            K.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.3)  # NUEVO: Borrado aleatorio
        ]
    return nn.Sequential(*aug_list).to(device)

def train_model(
    model, train_loader, val_loader, foul_criterion, action_criterion,
    num_epochs=200, device="cuda:0",
    use_focal_loss=False, use_mixup=False, use_cutmix=False, use_extra_aug=True,
    scheduler_type="onecycle"
):
    model = model.to(device)

    augment = get_augmentations(device, use_extra_aug=use_extra_aug) if "train" in train_loader.dataset.split else nn.Identity()

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=num_epochs * len(train_loader), pct_start=0.1)
    scaler = GradScaler()

    os.makedirs("models", exist_ok=True)
    best_val_ba = 0.0
    patience = 50
    patience_counter = 0
    accumulation_steps = 2

    # CAMBIO: Guardar los 3 mejores modelos
    top_models = []

    val_gt_foul_json = generate_groundtruth_json(val_loader.dataset, "Foul")
    val_gt_action_json = generate_groundtruth_json(val_loader.dataset, "Action")
    with open("val_gt_foul.json", "w") as f:
        json.dump(val_gt_foul_json, f)
    with open("val_gt_action.json", "w") as f:
        json.dump(val_gt_action_json, f)

    for epoch in range(num_epochs):
        if epoch == 2:
            print("Unfreezing the backbone...")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            if scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-epoch)
            else:
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=(num_epochs - epoch) * len(train_loader), pct_start=0.1)

        model.train()
        train_foul_loss = 0.0
        train_action_loss = 0.0
        all_foul_preds, all_action_preds, all_foul_labels, all_action_labels = [], [], [], []
        train_predictions = {}

        optimizer.zero_grad()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train MultiTaskMamba]") as pbar:
            for batch_idx, (batch_clips, foul_labels, action_labels, action_ids) in enumerate(pbar):
                batch_clips = batch_clips.to(device, non_blocking=True)
                foul_labels = foul_labels.to(device, non_blocking=True)
                action_labels = action_labels.to(device, non_blocking=True)

                B, V, C, T, H, W = batch_clips.shape
                batch_clips = batch_clips.reshape(B * V * T, C, H, W)
                batch_clips = augment(batch_clips)
                batch_clips = batch_clips.reshape(B, V, C, T, H, W)

                # CAMBIO: Mixup/Cutmix (solo si se activa)
                if use_mixup:
                    batch_clips, foul_labels_a, foul_labels_b, lam = mixup_data(batch_clips, foul_labels)
                    action_labels_a, action_labels_b, _ = mixup_data(batch_clips, action_labels)
                # (Cutmix no implementado aquí por simplicidad, pero puedes añadirlo similar a mixup)

                with autocast():
                    foul_logits, action_logits = model(batch_clips)
                    if use_focal_loss:
                        foul_loss = FocalLoss()(foul_logits, foul_labels)
                        action_loss = FocalLoss()(action_logits, action_labels)
                    elif use_mixup:
                        foul_loss = mixup_criterion(foul_criterion, foul_logits, foul_labels_a, foul_labels_b, lam)
                        action_loss = mixup_criterion(action_criterion, action_logits, action_labels_a, action_labels_b, lam)
                    else:
                        foul_loss = foul_criterion(foul_logits, foul_labels)
                        action_loss = action_criterion(action_logits, action_labels)
                    loss = (foul_loss + action_loss) / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_foul_loss += foul_loss.item()
                train_action_loss += action_loss.item()

                foul_preds = torch.argmax(foul_logits, 1)
                action_preds = torch.argmax(action_logits, 1)
                all_foul_preds.extend(foul_preds.cpu().tolist())
                all_action_preds.extend(action_preds.cpu().tolist())
                all_foul_labels.extend(foul_labels.cpu().tolist())
                all_action_labels.extend(action_labels.cpu().tolist())

                for action_id, f_logit, a_logit in zip(action_ids, foul_logits, action_logits):
                    f_probs = torch.softmax(f_logit, dim=0).detach().cpu().numpy()
                    a_probs = torch.softmax(a_logit, dim=0).detach().cpu().numpy()
                    f_pred = torch.argmax(f_logit).item()
                    a_pred = torch.argmax(a_logit).item()
                    f_label = {v: k for k, v in MVFoulDataset.foul_map.items()}[f_pred]
                    a_label = {v: k for k, v in MVFoulDataset.action_map.items()}[a_pred]
                    train_predictions[action_id] = {
                        "Offence": "No offence" if f_label == "No Offence" else "Offence",
                        "Severity": "" if f_label == "No Offence" else f_label.split("Severity ")[1],
                        "confidence_foul": float(f_probs[f_pred]),
                        "Action class": a_label,
                        "confidence_action": float(a_probs[a_pred])
                    }

                pbar.set_postfix({'foul_loss': train_foul_loss / (pbar.n + 1), 'action_loss': train_action_loss / (pbar.n + 1)})

                del batch_clips, foul_logits, action_logits, loss
                torch.cuda.empty_cache()

        train_foul_loss /= len(train_loader)
        train_action_loss /= len(train_loader)
        train_foul_acc = sum(p == l for p, l in zip(all_foul_preds, all_foul_labels)) / len(all_foul_labels)
        train_action_acc = sum(p == l for p, l in zip(all_action_preds, all_action_labels)) / len(all_action_labels)
        train_foul_ba = compute_balanced_accuracy(all_foul_labels, all_foul_preds, 4)
        train_action_ba = compute_balanced_accuracy(all_action_labels, all_action_preds, 8)
        train_foul_pred_counts = np.bincount(all_foul_preds, minlength=4)
        train_action_pred_counts = np.bincount(all_action_preds, minlength=8)
        train_foul_true_counts = np.bincount(all_foul_labels, minlength=4)
        train_action_true_counts = np.bincount(all_action_labels, minlength=8)

        train_pred_json = {"Actions": train_predictions}
        with open(f"train_pred_multitask_mamba_epoch{epoch+1}.json", "w") as f:
            json.dump(train_pred_json, f)

        torch.cuda.empty_cache()

        model.eval()
        val_foul_loss = 0.0
        val_action_loss = 0.0
        all_foul_preds, all_action_preds, all_foul_labels, all_action_labels = [], [], [], []
        val_predictions = {}

        with torch.no_grad():
            for batch_clips, foul_labels, action_labels, action_ids in val_loader:
                batch_clips = batch_clips.to(device, non_blocking=True)
                foul_labels = foul_labels.to(device, non_blocking=True)
                action_labels = action_labels.to(device, non_blocking=True)

                with autocast():
                    foul_logits, action_logits = model(batch_clips)
                    if use_focal_loss:
                        foul_loss = FocalLoss()(foul_logits, foul_labels)
                        action_loss = FocalLoss()(action_logits, action_labels)
                    else:
                        foul_loss = foul_criterion(foul_logits, foul_labels)
                        action_loss = action_criterion(action_logits, action_labels)

                val_foul_loss += foul_loss.item()
                val_action_loss += action_loss.item()

                foul_preds = torch.argmax(foul_logits, 1)
                action_preds = torch.argmax(action_logits, 1)
                all_foul_preds.extend(foul_preds.cpu().tolist())
                all_action_preds.extend(action_preds.cpu().tolist())
                all_foul_labels.extend(foul_labels.cpu().tolist())
                all_action_labels.extend(action_labels.cpu().tolist())

                for action_id, f_logit, a_logit in zip(action_ids, foul_logits, action_logits):
                    f_probs = torch.softmax(f_logit, dim=0).detach().cpu().numpy()
                    a_probs = torch.softmax(a_logit, dim=0).detach().cpu().numpy()
                    f_pred = torch.argmax(f_logit).item()
                    a_pred = torch.argmax(a_logit).item()
                    f_label = {v: k for k, v in MVFoulDataset.foul_map.items()}[f_pred]
                    a_label = {v: k for k, v in MVFoulDataset.action_map.items()}[a_pred]
                    val_predictions[action_id] = {
                        "Offence": "No offence" if f_label == "No Offence" else "Offence",
                        "Severity": "" if f_label == "No Offence" else f_label.split("Severity ")[1],
                        "confidence_foul": float(f_probs[f_pred]),
                        "Action class": a_label,
                        "confidence_action": float(a_probs[a_pred])
                    }

                del batch_clips, foul_logits, action_logits
                torch.cuda.empty_cache()

        val_foul_loss /= len(val_loader)
        val_action_loss /= len(val_loader)
        val_foul_acc = sum(p == l for p, l in zip(all_foul_preds, all_foul_labels)) / len(all_foul_labels)
        val_action_acc = sum(p == l for p, l in zip(all_action_preds, all_action_labels)) / len(all_action_labels)
        val_foul_ba = compute_balanced_accuracy(all_foul_labels, all_foul_preds, 4)
        val_action_ba = compute_balanced_accuracy(all_action_labels, all_action_preds, 8)
        val_foul_pred_counts = np.bincount(all_foul_preds, minlength=4)
        val_action_pred_counts = np.bincount(all_action_preds, minlength=8)
        val_foul_true_counts = np.bincount(all_foul_labels, minlength=4)
        val_action_true_counts = np.bincount(all_action_labels, minlength=8)

        val_pred_json = {"Actions": val_predictions}
        with open(f"val_pred_multitask_mamba_epoch{epoch+1}.json", "w") as f:
            json.dump(val_pred_json, f)

        foul_results = custom_evaluate(val_pred_json, val_gt_foul_json, "Foul")
        action_results = custom_evaluate(val_pred_json, val_gt_action_json, "Action")

        sn_foul_acc = foul_results["accuracy_offence_severity"]
        sn_foul_ba = foul_results["balanced_accuracy_offence_severity"]
        sn_foul_per_class = foul_results["per_class_offence"]
        sn_action_acc = action_results["accuracy_action"]
        sn_action_ba = action_results["balanced_accuracy_action"]
        sn_action_per_class = action_results["per_class_action"]

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Foul Loss: {train_foul_loss:.4f}, Train Action Loss: {train_action_loss:.4f}, "
              f"Train Foul Acc: {train_foul_acc:.4f}, Train Action Acc: {train_action_acc:.4f}, "
              f"Train Foul BA: {train_foul_ba:.4f}, Train Action BA: {train_action_ba:.4f}, "
              f"Val Foul Loss: {val_foul_loss:.4f}, Val Action Loss: {val_action_loss:.4f}, "
              f"Val Foul Acc: {val_foul_acc:.4f}, Val Action Acc: {val_action_acc:.4f}, "
              f"Val Foul BA: {val_foul_ba:.4f}, Val Action BA: {val_action_ba:.4f}")
        print(f"SoccerNet Foul Metrics - Val Acc: {sn_foul_acc:.4f}, Val BA: {sn_foul_ba:.4f}, Per-Class Acc: {sn_foul_per_class}")
        print(f"SoccerNet Action Metrics - Val Acc: {sn_action_acc:.4f}, Val BA: {sn_action_ba:.4f}, Per-Class Acc: {sn_action_per_class}")
        print(f"Train Foul true distribution: {train_foul_true_counts}")
        print(f"Train Foul pred distribution: {train_foul_pred_counts}")
        print(f"Train Action true distribution: {train_action_true_counts}")
        print(f"Train Action pred distribution: {train_action_pred_counts}")
        print(f"Val Foul true distribution: {val_foul_true_counts}")
        print(f"Val Foul pred distribution: {val_foul_pred_counts}")
        print(f"Val Action true distribution: {val_action_true_counts}")
        print(f"Val Action pred distribution: {val_action_pred_counts}")

        combined_ba = (sn_foul_ba + sn_action_ba) / 2
        # CAMBIO: Guardar los 3 mejores modelos
        if len(top_models) < 3 or combined_ba > min([m[0] for m in top_models]):
            torch.save(model.state_dict(), f"models/best_multitask_mamba_model_epoch{epoch+1}_ba{combined_ba:.4f}.pth")
            top_models.append((combined_ba, f"models/best_multitask_mamba_model_epoch{epoch+1}_ba{combined_ba:.4f}.pth"))
            top_models = sorted(top_models, key=lambda x: -x[0])[:3]
            print(f"Saved top MultiTaskMamba model with Combined SoccerNet Val BA: {combined_ba:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        torch.cuda.empty_cache()

if __name__ == "__main__":
    device = "cuda:0"

    train_data_dirs = ["/kaggle/input/datasetmvfd/datasetMVFD/train_preprocessed", "/kaggle/input/datasetmvfd/datasetMVFD/valid_preprocessed"]
    train_json_paths = ["/kaggle/input/datasetmvfd/datasetMVFD/train_preprocessed/annotations.json", "/kaggle/input/datasetmvfd/datasetMVFD/valid_preprocessed/annotations.json"]

    train_dataset = MVFoulDataset(
        train_data_dirs,
        train_json_paths,
        split='train',
        curriculum=True,
        preload=True,
        downsample_factor=2,
        max_clips_per_video=2
    )

    val_dataset = MVFoulDataset(
        "/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed",
        "/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed/annotations.json",
        split='val',
        preload=True,
        downsample_factor=2,
        max_clips_per_video=2
    )

    foul_weights, action_weights, train_foul_counts, train_action_counts = compute_class_weights(
        train_json_paths, train_dataset.foul_map, train_dataset.action_map
    )
    _, _, val_foul_counts, val_action_counts = compute_class_weights(
        "/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed/annotations.json", val_dataset.foul_map, val_dataset.action_map
    )

    print_unique_values_and_frequencies(train_dataset, "Training (Train+Valid)", train_foul_counts, train_action_counts)
    print_unique_values_and_frequencies(val_dataset, "Validation (Test)", val_foul_counts, val_action_counts)

    print(f"Training dataset size (original): {len(train_dataset.action_folders)}")
    print(f"Training dataset size (with curriculum): {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)

    print("\nTraining MultiTaskMamba Model...")
    multitask_model = MultiTaskModelMamba()
    # CAMBIO: Aumentar label smoothing a 0.1
    foul_criterion = nn.CrossEntropyLoss(weight=foul_weights.to(device), label_smoothing=0.1)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights.to(device), label_smoothing=0.1)
    # CAMBIO: Puedes activar focal loss, mixup, cutmix, augment extra y scheduler cosine aquí:
    train_model(
        multitask_model, train_loader, val_loader,
        foul_criterion, action_criterion,
        device=device,
        use_focal_loss=False,  # CAMBIO: pon True para usar focal loss
        use_mixup=False,       # CAMBIO: pon True para usar mixup
        use_cutmix=False,      # CAMBIO: pon True para usar cutmix (no implementado aquí)
        use_extra_aug=True,    # CAMBIO: pon False para solo augmentaciones básicas
        scheduler_type="onecycle"  # CAMBIO: pon "cosine" para CosineAnnealingLR
    )
