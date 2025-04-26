import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F  # Added for F.interpolate
import torch.optim as optim
from model_2 import SimpleFoulModel, SimpleActionModel, MultiTaskModel, MultiTaskModelMamba  # Updated import
from tqdm import tqdm
import numpy as np
import kornia.augmentation as K

class MVFoulDataset(Dataset):
    foul_map = {"No Offence": 0, "Offence Severity 1": 1, "Offence Severity 3": 2, "Offence Severity 5": 3}
    action_map = {
        "Standing Tackling": 0, "Tackling": 1, "Holding": 2, "Pushing": 3,
        "Challenge": 4, "Dive": 5, "High Leg": 6, "Elbowing": 7
    }

    def __init__(self, data_dirs, json_paths, num_frames=16, split='train', curriculum=False, preload=True):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.json_paths = json_paths if isinstance(json_paths, list) else [json_paths]
        self.metadata = {}
        for json_path in self.json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.metadata.update(data["Actions"])
        self.action_folders = []
        for data_dir in self.data_dirs:
            self.action_folders.extend([d for d in os.listdir(data_dir) if d.endswith(".pt")])
        self.num_frames = num_frames
        self.split = split
        self.curriculum = curriculum
        self.preload = preload
        self.action_normalization = {
            "standing tackle": "Standing Tackling", "tackle": "Tackling", "high leg": "High Leg",
            "dont know": None, "": None, "challenge": "Challenge", "dive": "Dive",
            "elbowing": "Elbowing", "holding": "Holding", "pushing": "Pushing", "high Leg": "High Leg"
        }
        
        self.valid_action_folders = []
        self.foul_labels = []
        self.action_labels = []
        self.folder_to_dir = {}
        
        for data_dir in self.data_dirs:
            for folder in os.listdir(data_dir):
                if not folder.endswith(".pt"):
                    continue
                action_id = folder.replace(".pt", "").replace("action_", "")
                if action_id not in self.metadata:
                    continue
                offence = self.metadata[action_id]["Offence"].lower()
                severity_str = self.metadata[action_id]["Severity"]
                action_class = self.metadata[action_id]["Action class"].lower()
                
                normalized_action = self.action_normalization.get(action_class, action_class.title())
                if normalized_action is None or normalized_action not in self.action_map:
                    continue
                
                if (offence == '' or offence == 'between') and normalized_action != 'Dive':
                    continue
                if (severity_str == '' or severity_str == '2.0' or severity_str == '4.0') and \
                   normalized_action != 'Dive' and offence != 'no offence':
                    continue
                
                if offence == '' or offence == 'between':
                    offence = 'offence'
                if severity_str == '' or severity_str == '2.0' or severity_str == '4.0':
                    severity_str = '1.0'
                
                if offence == 'no offence':
                    foul_label = 0
                elif offence == 'offence':
                    severity = float(severity_str)
                    if severity == 1.0:
                        foul_label = 1
                    elif severity == 3.0:
                        foul_label = 2
                    elif severity == 5.0:
                        foul_label = 3
                    else:
                        continue
                else:
                    continue
                
                action_label = self.action_map[normalized_action]
                
                self.valid_action_folders.append(folder)
                self.foul_labels.append(foul_label)
                self.action_labels.append(action_label)
                self.folder_to_dir[folder] = data_dir
        
        self.action_folders = self.valid_action_folders
        
        if self.preload:
            print(f"Preloading {len(self.action_folders)} .pt files for {split} split...")
            self.clips_cache = {}
            for folder in self.action_folders:
                action_id = folder.replace(".pt", "").replace("action_", "")
                action_path = os.path.join(self.folder_to_dir[folder], folder)
                clips = torch.load(action_path, weights_only=False).float() / 255.0
                if clips.shape[0] == 1:
                    clips = torch.stack([clips[0], clips[0]])
                else:
                    random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                    clips = torch.stack([clips[0], clips[random_idx]])
                self.clips_cache[action_id] = clips
        
        if self.curriculum and self.split == 'train':
            indices = []
            for i in range(len(self.action_folders)):
                action_factor = 1 if self.action_labels[i] not in [5, 6, 7] else (3 if self.action_labels[i] == 5 else 2)
                foul_factor = 5 if self.foul_labels[i] in [0, 3] else (2 if self.foul_labels[i] == 2 else 1)
                factor = min(max(action_factor, foul_factor), 5)
                indices.extend([i] * factor)
            self.indices = indices
            print(f"Number of samples after oversampling: {len(self.indices)}")
        else:
            self.indices = list(range(len(self.action_folders)))
            print(f"Number of samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        action_id = self.action_folders[actual_idx].replace(".pt", "").replace("action_", "")
        if self.preload:
            clips = self.clips_cache[action_id]
        else:
            action_path = os.path.join(self.folder_to_dir[self.action_folders[actual_idx]], self.action_folders[actual_idx])
            clips = torch.load(action_path, weights_only=False).float() / 255.0
            if clips.shape[0] == 1:
                clips = torch.stack([clips[0], clips[0]])
            else:
                random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                clips = torch.stack([clips[0], clips[random_idx]])
        return clips, torch.tensor(self.foul_labels[actual_idx]), torch.tensor(self.action_labels[actual_idx]), action_id

def custom_collate(batch):
    clips = torch.stack([item[0] for item in batch])
    foul_labels = torch.tensor([item[1].item() for item in batch], dtype=torch.long)
    action_labels = torch.tensor([item[2].item() for item in batch], dtype=torch.long)
    action_ids = [item[3] for item in batch]
    return clips, foul_labels, action_labels, action_ids

def compute_class_weights(json_paths, foul_map, action_map):
    metadata = {}
    for json_path in json_paths if isinstance(json_paths, list) else [json_paths]:
        with open(json_path, 'r') as f:
            data = json.load(f)
            metadata.update(data["Actions"])
    action_normalization = {
        "standing tackle": "Standing Tackling", "tackle": "Tackling", "high leg": "High Leg",
        "dont know": None, "": None, "challenge": "Challenge", "dive": "Dive",
        "elbowing": "Elbowing", "holding": "Holding", "pushing": "Pushing", "high Leg": "High Leg"
    }
    foul_labels = []
    action_labels = []
    for action_id, video in metadata.items():
        offence = video["Offence"].lower()
        severity_str = video["Severity"]
        action_class = video["Action class"].lower()
        normalized_action = action_normalization.get(action_class, action_class.title())
        if normalized_action is None or normalized_action not in action_map:
            continue
        if (offence == '' or offence == 'between') and normalized_action != 'Dive':
            continue
        if (severity_str == '' or severity_str == '2.0' or severity_str == '4.0') and \
           normalized_action != 'Dive' and offence != 'no offence':
            continue
        if offence == '' or offence == 'between':
            offence = 'offence'
        if severity_str == '' or severity_str == '2.0' or severity_str == '4.0':
            severity_str = '1.0'
        if offence == 'no offence':
            foul_label = 0
        elif offence == 'offence':
            severity = float(severity_str)
            if severity == 1.0:
                foul_label = 1
            elif severity == 3.0:
                foul_label = 2
            elif severity == 5.0:
                foul_label = 3
            else:
                continue
        else:
            continue
        action_label = action_map[normalized_action]
        foul_labels.append(foul_label)
        action_labels.append(action_label)
    foul_counts = torch.bincount(torch.tensor(foul_labels), minlength=len(foul_map))
    action_counts = torch.bincount(torch.tensor(action_labels), minlength=len(action_map))
    foul_weights = 1.0 / torch.sqrt(foul_counts.float() + 1e-6)
    action_weights = 1.0 / torch.sqrt(action_counts.float() + 1e-6)
    return foul_weights / foul_weights.sum(), action_weights / action_weights.sum(), foul_counts, action_counts

def print_unique_values_and_frequencies(dataset, split_name, foul_counts, action_counts):
    action_classes = set()
    severities = set()
    offences = set()
    for folder in dataset.action_folders:
        action_id = folder.replace(".pt", "").replace("action_", "")
        meta = dataset.metadata[action_id]
        action_classes.add(meta["Action class"])
        severities.add(meta["Severity"] if meta["Severity"] else "Empty")
        offences.add(meta["Offence"] if meta["Offence"] else "Empty")
    print(f"\nUnique values for {split_name} split:")
    print(f"Action classes: {sorted(action_classes)}")
    print(f"Severities: {sorted(severities)}")
    print(f"Offences: {sorted(offences)}")
    foul_labels_map = {0: "No Offence", 1: "Offence Severity 1", 2: "Offence Severity 3", 3: "Offence Severity 5"}
    print(f"\nFoul label frequencies for {split_name} split:")
    for label, count in enumerate(foul_counts):
        print(f"{foul_labels_map[label]}: {int(count)}")
    action_labels_map = {v: k for k, v in MVFoulDataset.action_map.items()}
    print(f"\nAction label frequencies for {split_name} split:")
    for label, count in enumerate(action_counts):
        print(f"{action_labels_map[label]}: {int(count)}")

def generate_predictions_json(action_ids, logits, task_name):
    predictions = {"Actions": {}}
    reverse_foul_map = {v: k for k, v in MVFoulDataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in MVFoulDataset.action_map.items()}
    
    for action_id, logit in zip(action_ids, logits):
        probs = torch.softmax(logit, dim=0).detach().cpu().numpy()
        pred_idx = torch.argmax(logit).item()
        if task_name == "Foul":
            pred_label = reverse_foul_map[pred_idx]
            if pred_label == "No Offence":
                offence = "No offence"
                severity = ""
            else:
                offence = "Offence"
                severity = pred_label.split("Severity ")[1]
            predictions["Actions"][action_id] = {
                "Offence": offence,
                "Severity": severity,
                "confidence": float(probs[pred_idx])
            }
        else:  # Action
            pred_label = reverse_action_map[pred_idx]
            predictions["Actions"][action_id] = {
                "Action class": pred_label,
                "confidence": float(probs[pred_idx])
            }
    return predictions

def generate_groundtruth_json(dataset, task_name):
    groundtruth = {"Actions": {}}
    reverse_foul_map = {v: k for k, v in dataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in dataset.action_map.items()}
    
    for idx in range(len(dataset.action_folders)):
        action_id = dataset.action_folders[idx].replace(".pt", "").replace("action_", "")
        action_class = reverse_action_map[dataset.action_labels[idx]]
        if task_name == "Foul":
            foul_label = reverse_foul_map[dataset.foul_labels[idx]]
            if foul_label == "No Offence":
                offence = "No offence"
                severity = ""
            else:
                offence = "Offence"
                severity = foul_label.split("Severity ")[1]
            groundtruth["Actions"][action_id] = {
                "Action class": action_class,
                "Offence": offence,
                "Severity": severity
            }
        else:  # Action
            action_label = reverse_action_map[dataset.action_labels[idx]]
            groundtruth["Actions"][action_id] = {
                "Action class": action_label
            }
    return groundtruth

def custom_evaluate(predictions, groundtruth, task_name):
    if task_name == "Foul":
        num_classes = 4
        class_names = ["No offence", "Offence Severity 1", "Offence Severity 3", "Offence Severity 5"]
    else:  # Action
        num_classes = 8
        class_names = list(MVFoulDataset.action_map.keys())
    
    true_counts = np.zeros(num_classes)
    pred_correct = np.zeros(num_classes)
    
    for action_id in groundtruth["Actions"]:
        true_action = groundtruth["Actions"][action_id]["Action class"]
        if task_name == "Foul":
            true_offence = groundtruth["Actions"][action_id]["Offence"]
            true_severity = groundtruth["Actions"][action_id]["Severity"]
            if true_offence == "No offence":
                true_idx = 0
            elif true_offence == "Offence":
                if true_severity == "1":
                    true_idx = 1
                elif true_severity == "3":
                    true_idx = 2
                elif true_severity == "5":
                    true_idx = 3
                else:
                    continue
            else:
                continue
            true_counts[true_idx] += 1
        else:
            true_idx = MVFoulDataset.action_map[true_action]
            true_counts[true_idx] += 1
        
        if action_id in predictions["Actions"]:
            if task_name == "Foul":
                pred_offence = predictions["Actions"][action_id]["Offence"]
                pred_severity = predictions["Actions"][action_id]["Severity"]
                if pred_offence == true_offence and pred_severity == true_severity:
                    pred_correct[true_idx] += 1
            else:
                pred_action = predictions["Actions"][action_id]["Action class"]
                if pred_action == true_action:
                    pred_correct[true_idx] += 1
    
    accuracy = sum(pred_correct) / sum(true_counts) if sum(true_counts) > 0 else 0.0
    per_class_acc = {}
    for i, name in enumerate(class_names):
        per_class_acc[name] = pred_correct[i] / true_counts[i] if true_counts[i] > 0 else 0.0
    ba = np.mean(pred_correct / true_counts) if sum(true_counts) > 0 else 0.0
    
    if task_name == "Foul":
        return {
            "accuracy_offence_severity": accuracy * 100,
            "balanced_accuracy_offence_severity": ba * 100,
            "per_class_offence": per_class_acc
        }
    else:
        return {
            "accuracy_action": accuracy * 100,
            "balanced_accuracy_action": ba * 100,
            "per_class_action": per_class_acc
        }

def compute_balanced_accuracy(true_labels, pred_labels, num_classes):
    per_class_acc = []
    for cls in range(num_classes):
        cls_true = [1 if t == cls else 0 for t in true_labels]
        cls_pred = [1 if p == cls else 0 for p in pred_labels]
        correct = sum(1 for t, p in zip(cls_true, cls_pred) if t == 1 and p == 1)
        total = sum(cls_true)
        acc = correct / total if total > 0 else 0.0
        per_class_acc.append(acc)
    return sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

def train_model(model, train_loader, val_loader, foul_criterion, action_criterion, num_epochs=200, device="cuda:0"):
    model = model.to(device)
    
    augment = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5)
    ).to(device) if "train" in train_loader.dataset.split else nn.Identity()
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=num_epochs * len(train_loader), pct_start=0.1)
    
    os.makedirs("models", exist_ok=True)
    best_val_ba = 0.0
    patience = 50
    patience_counter = 0
    
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
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=(num_epochs - epoch) * len(train_loader), pct_start=0.1)
        
        model.train()
        train_foul_loss = 0.0
        train_action_loss = 0.0
        all_foul_preds, all_action_preds, all_foul_labels, all_action_labels = [], [], [], []
        train_predictions = {}
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train MultiTaskMamba]") as pbar:
            for batch_clips, foul_labels, action_labels, action_ids in pbar:
                batch_clips = batch_clips.to(device, non_blocking=True)
                foul_labels = foul_labels.to(device, non_blocking=True)
                action_labels = action_labels.to(device, non_blocking=True)
                
                B, V, C, T, H, W = batch_clips.shape
                batch_clips = batch_clips.reshape(B * V * T, C, H, W)
                batch_clips = augment(batch_clips)
                batch_clips = batch_clips.reshape(B, V, C, T, H, W)
                
                optimizer.zero_grad()
                foul_logits, action_logits = model(batch_clips)
                
                foul_loss = foul_criterion(foul_logits, foul_labels)
                action_loss = action_criterion(action_logits, action_labels)
                loss = foul_loss + action_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
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
                
                foul_logits, action_logits = model(batch_clips)
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
        if combined_ba > best_val_ba:
            best_val_ba = combined_ba
            torch.save(model.state_dict(), "models/best_multitask_mamba_model.pth")
            print(f"Saved best MultiTaskMamba model with Combined SoccerNet Val BA: {best_val_ba:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    device = "cuda:0"
    
    train_data_dirs = ["dataset/train_preprocessed", "dataset/valid_preprocessed"]
    train_json_paths = ["dataset/train/annotations.json", "dataset/valid/annotations.json"]
    train_dataset = MVFoulDataset(train_data_dirs, train_json_paths, split='train', curriculum=True, preload=True)
    
    val_dataset = MVFoulDataset("dataset/test_preprocessed", "dataset/test/annotations.json", split='val', preload=True)
    
    foul_weights, action_weights, train_foul_counts, train_action_counts = compute_class_weights(
        train_json_paths, train_dataset.foul_map, train_dataset.action_map
    )
    _, _, val_foul_counts, val_action_counts = compute_class_weights(
        "dataset/test/annotations.json", val_dataset.foul_map, val_dataset.action_map
    )
    
    print_unique_values_and_frequencies(train_dataset, "Training (Train+Valid)", train_foul_counts, train_action_counts)
    print_unique_values_and_frequencies(val_dataset, "Validation (Test)", val_foul_counts, val_action_counts)
    
    print(f"Training dataset size (original): {len(train_dataset.action_folders)}")
    print(f"Training dataset size (with curriculum): {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)
    
    print("\nTraining MultiTaskMamba Model...")
    multitask_model = MultiTaskModelMamba()  # Instantiate the new Mamba model
    foul_criterion = nn.CrossEntropyLoss(weight=foul_weights.to(device), label_smoothing=0.05)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights.to(device), label_smoothing=0.05)
    train_model(multitask_model, train_loader, val_loader, foul_criterion, action_criterion, device=device)
