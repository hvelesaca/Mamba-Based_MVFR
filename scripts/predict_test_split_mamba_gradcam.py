import os
import torch
from torch.utils.data import Dataset, DataLoader
from model_2_v4 import MultiTaskModelMamba
from tqdm import tqdm
import json
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, top_k_accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MVFoulTestDataset(Dataset):
    foul_map = {"No Offence": 0, "Offence Severity 1": 1, "Offence Severity 3": 2, "Offence Severity 5": 3}
    action_map = {"Standing Tackling": 0, "Tackling": 1, "Holding": 2, "Pushing": 3,
        "Challenge": 4, "Dive": 5, "High Leg": 6, "Elbowing": 7
    }

    def __init__(self, data_dir, json_path, num_frames=16, split='test', preload=True, filter_data=True):
        self.data_dir = data_dir
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.metadata = data["Actions"]
        self.action_folders = [d for d in os.listdir(data_dir) if d.endswith(".pt")]
        self.num_frames = num_frames
        self.split = split
        self.preload = preload
        self.filter_data = filter_data
        self.action_normalization = {
            "standing tackle": "Standing Tackling", "tackle": "Tackling", "high leg": "High Leg",
            "dont know": None, "": None, "challenge": "Challenge", "dive": "Dive",
            "elbowing": "Elbowing", "holding": "Holding", "pushing": "Pushing", "high Leg": "High Leg"
        }
        
        self.valid_action_folders = []
        self.foul_labels = []
        self.action_labels = []
        
        for folder in self.action_folders:
            action_id = folder.replace(".pt", "").replace("action_", "")
            if action_id not in self.metadata:
                if self.filter_data:
                    continue
                else:
                    # For whole dataset, include all actions even without metadata
                    self.valid_action_folders.append(folder)
                    self.foul_labels.append(-1)  # Dummy label
                    self.action_labels.append(-1)  # Dummy label
                    continue
            
            offence = self.metadata[action_id]["Offence"].lower()
            severity_str = self.metadata[action_id]["Severity"]
            action_class = self.metadata[action_id]["Action class"].lower()
            
            normalized_action = self.action_normalization.get(action_class, action_class.title())
            
            if self.filter_data:
                # Apply filtering only if filter_data is True
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
            else:
                # For whole dataset, include all actions with dummy labels if filtering is off
                foul_label = -1  # Dummy label
                action_label = -1  # Dummy label
                if normalized_action in self.action_map:
                    action_label = self.action_map[normalized_action]
            
            self.valid_action_folders.append(folder)
            self.foul_labels.append(foul_label)
            self.action_labels.append(action_label)
        
        self.action_folders = self.valid_action_folders
        print(f"Total .pt files found: {len(self.action_folders)}")
        
        if self.preload:
            print(f"Preloading {len(self.action_folders)} .pt files for {split} split...")
            self.clips_cache = {}
            for folder in self.action_folders:
                action_path = os.path.join(self.data_dir, folder)
                clips = torch.load(action_path, weights_only=False).float() / 255.0
                if clips.shape[0] == 1:
                    clips = torch.stack([clips[0], clips[0]])
                else:
                    random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                    clips = torch.stack([clips[0], clips[random_idx]])
                action_id = folder.replace(".pt", "").replace("action_", "")
                self.clips_cache[action_id] = clips
        
        self.indices = list(range(len(self.action_folders)))
        print(f"Number of samples after processing: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        action_id = self.action_folders[actual_idx].replace(".pt", "").replace("action_", "")
        
        if self.preload:
            clips = self.clips_cache[action_id]
        else:
            action_path = os.path.join(self.data_dir, self.action_folders[actual_idx])
            clips = torch.load(action_path, weights_only=False).float() / 255.0
            if clips.shape[0] == 1:
                clips = torch.stack([clips[0], clips[0]])
            else:
                random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                clips = torch.stack([clips[0], clips[random_idx]])
        
        foul_label = self.foul_labels[actual_idx]
        action_label = self.action_labels[actual_idx]
        
        return clips, torch.tensor(foul_label), torch.tensor(action_label), action_id


class MVFoulTestDataset2(Dataset):
    foul_map = {
        "No Offence": 0,
        "Offence Severity 1": 1,
        "Offence Severity 3": 2,
        "Offence Severity 5": 3
    }
    action_map = {
        "Standing Tackling": 0, "Tackling": 1, "Holding": 2, "Pushing": 3,
        "Challenge": 4, "Dive": 5, "High Leg": 6, "Elbowing": 7
    }
    action_normalization = {
        "standing tackle": "Standing Tackling", "tackle": "Tackling", "high leg": "High Leg",
        "dont know": None, "": None, "challenge": "Challenge", "dive": "Dive",
        "elbowing": "Elbowing", "holding": "Holding", "pushing": "Pushing", "high Leg": "High Leg"
    }

    def __init__(self, data_dir, json_path, num_frames=16, split='test', preload=True):
        self.data_dir = data_dir
        self.json_path = json_path
        self.num_frames = num_frames
        self.split = split
        self.preload = preload
        
        # Load metadata
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.metadata = data["Actions"]
        
        # Check if data_dir exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}. Please check the data_dir path.")
        
        print(f"Checking directory: {data_dir}")
        print(f"Contents of {data_dir}: {os.listdir(data_dir)}")
        self.action_folders = [d for d in os.listdir(data_dir) if d.endswith(".pt")]
        
        if not self.action_folders:
            print(f"Warning: No .pt files found in {data_dir}. Dataset is empty.")
        else:
            print(f"First few .pt files: {self.action_folders[:5]}")
        
        self.action_folders.sort()  # Ensure consistent order
        print(f"Total .pt files found: {len(self.action_folders)}")
        
        # Filter valid actions
        self.valid_action_folders = []
        self.foul_labels = []
        self.action_labels = []
        
        for folder in self.action_folders:
            action_id = folder.replace(".pt", "").replace("action_", "")
            if action_id not in self.metadata:
                print(f"Warning: Action ID {action_id} not in metadata, skipping")
                continue
            
            offence = self.metadata[action_id]["Offence"].lower()
            severity_str = self.metadata[action_id]["Severity"]
            action_class = self.metadata[action_id]["Action class"].lower()
            
            normalized_action = self.action_normalization.get(action_class, action_class.title())
            
            # Filtering logic
            if normalized_action is None or normalized_action not in self.action_map:
                print(f"Warning: Invalid action class '{action_class}' for action {action_id}, skipping")
                continue
            
            if (offence == '' or offence == 'between') and normalized_action != 'Dive':
                print(f"Warning: Invalid offence '{offence}' for action {action_id}, skipping")
                continue
            if (severity_str == '' or severity_str == '2.0' or severity_str == '4.0') and \
               normalized_action != 'Dive' and offence != 'no offence':
                severity_str = '1.0'  # Assign default severity
                print(f"Assigned default severity 1.0 for action {action_id}")
            
            if offence == '' or offence == 'between':
                offence = 'offence'
            
            if offence == 'no offence':
                foul_label = 0
            elif offence == 'offence':
                try:
                    severity = float(severity_str)
                    if severity == 1.0:
                        foul_label = 1
                    elif severity == 3.0:
                        foul_label = 2
                    elif severity == 5.0:
                        foul_label = 3
                    else:
                        print(f"Warning: Unsupported severity {severity} for action {action_id}, skipping")
                        continue
                except ValueError:
                    print(f"Warning: Invalid severity '{severity_str}' for action {action_id}, skipping")
                    continue
            else:
                print(f"Warning: Invalid offence '{offence}' for action {action_id}, skipping")
                continue
            
            action_label = self.action_map[normalized_action]
            
            self.valid_action_folders.append(folder)
            self.foul_labels.append(foul_label)
            self.action_labels.append(action_label)
        
        self.action_folders = self.valid_action_folders
        print(f"Total valid .pt files after filtering: {len(self.action_folders)}")
        
        if self.preload:
            print(f"Preloading {len(self.action_folders)} .pt files for {split} split...")
            self.clips_cache = {}
            for folder in self.action_folders:
                action_path = os.path.join(self.data_dir, folder)
                clips = torch.load(action_path, weights_only=False).float() / 255.0
                if clips.shape[0] == 1:
                    clips = torch.stack([clips[0], clips[0]])
                else:
                    random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                    clips = torch.stack([clips[0], clips[random_idx]])
                action_id = folder.replace(".pt", "").replace("action_", "")
                self.clips_cache[action_id] = clips
        
        self.indices = list(range(len(self.action_folders)))
        print(f"Number of samples after processing: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        action_id = self.action_folders[actual_idx].replace(".pt", "").replace("action_", "")
        
        if self.preload:
            clips = self.clips_cache[action_id]
        else:
            action_path = os.path.join(self.data_dir, self.action_folders[actual_idx])
            clips = torch.load(action_path, weights_only=False).float() / 255.0
            if clips.shape[0] == 1:
                clips = torch.stack([clips[0], clips[0]])
            else:
                random_idx = torch.randint(1, clips.shape[0], (1,)).item()
                clips = torch.stack([clips[0], clips[random_idx]])
        
        foul_label = self.foul_labels[actual_idx]
        action_label = self.action_labels[actual_idx]
        
        return clips, torch.tensor(foul_label), torch.tensor(action_label), action_id

def custom_collate(batch):
    clips = torch.stack([item[0] for item in batch])
    foul_labels = torch.tensor([item[1].item() for item in batch], dtype=torch.long)
    action_labels = torch.tensor([item[2].item() for item in batch], dtype=torch.long)
    action_ids = [item[3] for item in batch]
    return clips, foul_labels, action_labels, action_ids

def generate_predictions_json(action_ids, foul_logits, action_logits):
    predictions = {"Actions": {}}
    reverse_foul_map = {v: k for k, v in MVFoulTestDataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in MVFoulTestDataset.action_map.items()}
    
    for action_id, foul_logit, action_logit in zip(action_ids, foul_logits, action_logits):
        foul_probs = torch.softmax(foul_logit, dim=0).detach().cpu().numpy()
        foul_pred_idx = torch.argmax(foul_logit).item()
        foul_label = reverse_foul_map[foul_pred_idx]
        if foul_label == "No Offence":
            offence = "No offence"
            severity = ""
        else:
            offence = "Offence"
            severity = foul_label.split("Severity ")[1] + ".0"
        
        action_probs = torch.softmax(action_logit, dim=0).detach().cpu().numpy()
        action_pred_idx = torch.argmax(action_logit).item()
        action_label = reverse_action_map[action_pred_idx]
        
        predictions["Actions"][action_id] = {
            "Action class": action_label,
            "confidence_action": float(action_probs[action_pred_idx]),
            "Offence": offence,
            "Severity": severity,
            "confidence_offence": float(foul_probs[foul_pred_idx]),
            "foul_pred_idx": foul_pred_idx,
            "action_pred_idx": action_pred_idx,
            "foul_probs": foul_probs.tolist(),
            "action_probs": action_probs.tolist()
        }
    return predictions

def compute_performance(dataset, predictions, save_dir="results"):
    foul_true, foul_pred, foul_probs = [], [], []
    action_true, action_pred, action_probs = [], [], []
    reverse_foul_map = {v: k for k, v in dataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in dataset.action_map.items()}
    
    print("Sample predictions:", list(predictions["Actions"].items())[:2])
    
    for idx, folder in enumerate(dataset.action_folders):
        action_id = folder.replace(".pt", "").replace("action_", "")
        if action_id not in predictions["Actions"]:
            print(f"Warning: Action ID {action_id} not in predictions")
            continue
        
        pred = predictions["Actions"][action_id]
        foul_label = dataset.foul_labels[idx]
        action_label = dataset.action_labels[idx]
        
        # Foul labels and probabilities
        if foul_label >= 0:
            foul_true.append(foul_label)
            foul_pred.append(pred["foul_pred_idx"])
            foul_probs.append(pred["foul_probs"])
        else:
            print(f"Warning: Invalid foul label for action {action_id}, skipping")
        
        # Action labels and probabilities
        if action_label >= 0:
            action_true.append(action_label)
            action_pred.append(pred["action_pred_idx"])
            action_probs.append(pred["action_probs"])
        else:
            print(f"Warning: Invalid action label for action {action_id}, skipping")
    
    foul_probs = np.array(foul_probs)
    action_probs = np.array(action_probs)
    
    # Compute metrics for foul classification
    if not foul_true:
        print("Warning: No valid foul labels processed")
        foul_metrics = {
            "Accuracy": 0.0, "Balanced Accuracy": 0.0, "F1": 0.0, "Recall": 0.0, "Precision": 0.0,
            "Top-2 Accuracy": 0.0, "Class Accuracies": {}
        }
    else:
        foul_accuracy = accuracy_score(foul_true, foul_pred)
        foul_balanced_accuracy = balanced_accuracy_score(foul_true, foul_pred)
        foul_f1 = f1_score(foul_true, foul_pred, average='weighted')
        foul_recall = recall_score(foul_true, foul_pred, average='weighted')
        foul_precision = precision_score(foul_true, foul_pred, average='weighted')
        foul_top2 = top_k_accuracy_score(foul_true, foul_probs, k=2, labels=list(range(len(dataset.foul_map))))
        
        # Per-class accuracies
        foul_cm = confusion_matrix(foul_true, foul_pred, labels=list(range(len(dataset.foul_map))))
        foul_class_accuracies = {}
        for i in range(len(dataset.foul_map)):
            true_positives = foul_cm[i, i]
            total = foul_cm[i, :].sum()
            class_acc = true_positives / total if total > 0 else 0.0
            foul_class_accuracies[reverse_foul_map[i]] = class_acc
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(foul_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(reverse_foul_map.values()),
                    yticklabels=list(reverse_foul_map.values()))
        plt.title("Foul Classification Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        foul_cm_path = os.path.join(save_dir, "foul_confusion_matrix.png")
        plt.savefig(foul_cm_path)
        plt.close()
        print(f"Saved foul confusion matrix to {foul_cm_path}")
        
        foul_metrics = {
            "Accuracy": foul_accuracy,
            "Balanced Accuracy": foul_balanced_accuracy,
            "F1": foul_f1,
            "Recall": foul_recall,
            "Precision": foul_precision,
            "Top-2 Accuracy": foul_top2,
            "Class Accuracies": foul_class_accuracies
        }
    
    # Compute metrics for action classification
    if not action_true:
        print("Warning: No valid action labels processed")
        action_metrics = {
            "Accuracy": 0.0, "Balanced Accuracy": 0.0, "F1": 0.0, "Recall": 0.0, "Precision": 0.0,
            "Top-2 Accuracy": 0.0, "Class Accuracies": {}
        }
    else:
        action_accuracy = accuracy_score(action_true, action_pred)
        action_balanced_accuracy = balanced_accuracy_score(action_true, action_pred)
        action_f1 = f1_score(action_true, action_pred, average='weighted')
        action_recall = recall_score(action_true, action_pred, average='weighted')
        action_precision = precision_score(action_true, action_pred, average='weighted')
        action_top2 = top_k_accuracy_score(action_true, action_probs, k=2, labels=list(range(len(dataset.action_map))))
        
        # Per-class accuracies
        action_cm = confusion_matrix(action_true, action_pred, labels=list(range(len(dataset.action_map))))
        action_class_accuracies = {}
        for i in range(len(dataset.action_map)):
            true_positives = action_cm[i, i]
            total = action_cm[i, :].sum()
            class_acc = true_positives / total if total > 0 else 0.0
            action_class_accuracies[reverse_action_map[i]] = class_acc
        
        # Confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(action_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(reverse_action_map.values()),
                    yticklabels=list(reverse_action_map.values()))
        plt.title("Action Classification Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        action_cm_path = os.path.join(save_dir, "action_confusion_matrix.png")
        plt.savefig(action_cm_path)
        plt.close()
        print(f"Saved action confusion matrix to {action_cm_path}")
        
        action_metrics = {
            "Accuracy": action_accuracy,
            "Balanced Accuracy": action_balanced_accuracy,
            "F1": action_f1,
            "Recall": action_recall,
            "Precision": action_precision,
            "Top-2 Accuracy": action_top2,
            "Class Accuracies": action_class_accuracies
        }
    
    # Print metrics
    print(f"Foul Metrics: {foul_metrics}")
    print(f"Action Metrics: {action_metrics}")
    
    # Save metrics to a text file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(save_dir, f"metrics_multitask_mamba_test_{timestamp}.txt")
    with open(metrics_file, 'w') as f:
        f.write("Foul Classification Metrics:\n")
        for key, value in foul_metrics.items():
            if key != "Class Accuracies":
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write("Per-Class Accuracies:\n")
                for cls, acc in value.items():
                    f.write(f"  {cls}: {acc:.4f}\n")
        f.write("\nAction Classification Metrics:\n")
        for key, value in action_metrics.items():
            if key != "Class Accuracies":
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write("Per-Class Accuracies:\n")
                for cls, acc in value.items():
                    f.write(f"  {cls}: {acc:.4f}\n")
    print(f"Saved metrics to {metrics_file}")
    
    return {
        "Foul": foul_metrics,
        "Action": action_metrics
    }

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

         
    def __call__(self, x, class_idx=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        B, V, C, T, H, W = x.shape
        x_flat = x.view(-1, C, T, H, W)  # [B*V, C, T, H, W]
        print(f"GradCAM input shape: {x_flat.shape}")
        
        foul_logits, action_logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(foul_logits, dim=1)
        if isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.to(foul_logits.device)
            
        self.model.zero_grad()
        score = foul_logits[:, class_idx].sum()
        score.backward()
        
        # Compute Grad-CAM
        B_flat, C_act, T_act, H_act, W_act = self.activations.shape
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        
        #B_flat, C_act, T_act, H_act, W_act = self.activations.shape
        print(f"Activations shape: {self.activations.shape}")
        print(f"Gradients shape: {self.gradients.shape}")
        print(f"Using T_act={T_act}, H_act={H_act}, W_act={W_act} for reshape")
        
        # Reshape cam to [B, V, T', H', W']
        cam = cam.view(B, V, T_act, H_act, W_act)
        return cam, foul_logits, action_logits

def visualize_gradcam(model, clips, action_ids, num_samples=15, num_views=2, save_dir="gradcam_visualizations"):
    try:
        print(f"Generating Grad-CAM visualizations for {min(num_samples, len(clips))} samples...")
        print(f"Input clips shape: {clips.shape}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Visualization directory: {save_dir}")

        if isinstance(model, torch.nn.DataParallel):
            backbone = model.module.backbone
        else:
            backbone = model.backbone

        if not hasattr(backbone, 'conv_proj'):
            print("Error: 'conv_proj' not found in model.backbone.")
            raise AttributeError("'conv_proj' not found in model.backbone")

        clips = clips[:num_samples]
        action_ids = action_ids[:num_samples]
        print(f"Selected clips shape: {clips.shape}")

        device = next(model.parameters()).device
        clips = clips.to(device, non_blocking=True).requires_grad_(True)

        resize = T.Resize((224, 224), antialias=True)
        resized_clips = torch.zeros(clips.shape[0], clips.shape[1], clips.shape[2], clips.shape[3], 224, 224,
                                  dtype=clips.dtype, device=device)
        for b in range(clips.shape[0]):
            for v in range(clips.shape[1]):
                for t in range(clips.shape[3]):
                    frame = clips[b, v, :, t, :, :]
                    resized_frame = resize(frame)
                    resized_clips[b, v, :, t, :, :] = resized_frame

        clips = resized_clips
        print(f"Resized clips shape: {clips.shape}")

        gradcam = GradCAM(model, backbone.conv_proj)
        cams, foul_logits, action_logits = gradcam(clips)
        print(f"Grad-CAM output shape: {cams.shape}")

        reverse_foul_map = {v: k for k, v in MVFoulTestDataset.foul_map.items()}
        reverse_action_map = {v: k for k, v in MVFoulTestDataset.action_map.items()}

        original_frame_indices = list(range(6, 11))
        gradcam_frame_indices = [i // 2 for i in original_frame_indices]

        for i in range(len(clips)):
            for v in range(num_views):
                clip_name = "Clip 0" if v == 0 else "Clip Random"
                cam = cams[i, v].detach().cpu().numpy()
                clip = clips[i, v].detach().cpu().numpy().transpose(1, 2, 3, 0)
                action_id = action_ids[i]
                foul_pred_idx = torch.argmax(foul_logits[i]).item()
                action_pred_idx = torch.argmax(action_logits[i]).item()

                selected_frames = clip[original_frame_indices]
                selected_cams = cam[gradcam_frame_indices]

                cam_resized = np.zeros((len(gradcam_frame_indices), clip.shape[1], clip.shape[2]))
                for t in range(len(gradcam_frame_indices)):
                    cam_t = cv2.resize(selected_cams[t], (clip.shape[2], clip.shape[1]))
                    cam_resized[t] = cam_t

                target_size = (398, 224)
                
                # Create heatmap without whitespace
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                for t in range(5):
                    frame = selected_frames[t]
                    frame = (frame - frame.min()) / (frame.max() - frame.min())
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

                    axes[0, t].imshow(frame)
                    axes[0, t].set_title(f"Frame {original_frame_indices[t]+1}")
                    axes[0, t].axis('off')

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized[t]), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
                    overlay = frame * 0.5 + heatmap * 0.5
                    axes[1, t].imshow(overlay)
                    axes[1, t].set_title(f"Grad-CAM {original_frame_indices[t]+1}")
                    axes[1, t].axis('off')

                    # Crear carpetas para cada imagen
                    image_dir = os.path.join(save_dir, f"action_{action_id}", f"view_{v}", f"frame_{t+1}")
                    os.makedirs(image_dir, exist_ok=True)

                    # Guardar frame y overlay                    
                    frame_path = os.path.join(image_dir, "frame.png")
                    overlay_path = os.path.join(image_dir, "overlay.png")
                   
                    cv2.imwrite(frame_path, (frame * 255).astype(np.uint8))
                    cv2.imwrite(overlay_path, (overlay * 255).astype(np.uint8))

                    frame = cv2.imread(frame_path) 
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    overlay = cv2.imread(overlay_path) 
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(frame_path, frame)

                    print(f"Saved frame: {frame_path}")
                    print(f"Saved overlay: {overlay_path}")

                plt.suptitle(f"Action ID: {action_id} - {clip_name}; Foul: {reverse_foul_map[foul_pred_idx]}; Action: {reverse_action_map[action_pred_idx]}")
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"gradcam_action_{action_id}_view_{v}.png")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved visualization: {save_path}")
                plt.close()
    except Exception as e:
        print(f"Error in visualize_gradcam: {str(e)}")

def predict_test_split(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    test_predictions = {"Actions": {}}
    all_clips = []
    all_action_ids = []
    
    with torch.no_grad():
        for batch_clips, foul_labels, action_labels, action_ids in tqdm(test_loader, desc="Predicting on Test Set", unit="batch"):
            # Resize clips to 224x224
            resize = T.Resize((224, 224), antialias=True)
            resized_clips = torch.zeros(batch_clips.shape[0], batch_clips.shape[1], batch_clips.shape[2], 
                                      batch_clips.shape[3], 224, 224, dtype=batch_clips.dtype, device=batch_clips.device)
            for b in range(batch_clips.shape[0]):
                for v in range(batch_clips.shape[1]):
                    for t in range(batch_clips.shape[3]):
                        frame = batch_clips[b, v, :, t, :, :]  # [C, H, W]
                        resized_frame = resize(frame)  # [C, 224, 224]
                        resized_clips[b, v, :, t, :, :] = resized_frame
            batch_clips = resized_clips.to(device, non_blocking=True)
            print(f"Batch clips shape after resize: {batch_clips.shape}")
            
            foul_logits, action_logits = model(batch_clips)
            
            batch_predictions = generate_predictions_json(action_ids, foul_logits, action_logits)
            test_predictions["Actions"].update(batch_predictions["Actions"])
            
            # Collect clips and action IDs for visualization
            all_clips.append(batch_clips.cpu())
            all_action_ids.extend(action_ids)
    
    # Concatenate all clips
    all_clips = torch.cat(all_clips, dim=0)  # [N, V, C, T, H, W]
    print(f"All clips shape: {all_clips.shape}")
    print(f"Total action IDs collected: {len(all_action_ids)}")

    print("all_action_ids: ", all_action_ids)
    # Randomly select 15 samples
    num_samples_to_visualize = 1
    if len(all_action_ids) < num_samples_to_visualize:
        print(f"Warning: Only {len(all_action_ids)} samples available, visualizing all.")
        num_samples_to_visualize = len(all_action_ids)

    #idx_selected = range(220, 230)
    #indices = np.random.choice(len(all_action_ids), num_samples_to_visualize, replace=False)
    idx_selected = [133, 91, 0, 58, 13, 250, 247, 20]
   
    for idx in idx_selected: 
        indices = [idx]
        selected_clips = all_clips[indices].to(device)
        selected_action_ids = [all_action_ids[i] for i in indices]
        print(f"Selected {num_samples_to_visualize} samples for visualization: {selected_action_ids}")
        
        # Visualize Grad-CAM for the selected samples
        with torch.enable_grad():
            visualize_gradcam(model, selected_clips, selected_action_ids, num_samples=num_samples_to_visualize)
    
    # Save predictions
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_file = f"results/predictions_multitask_mamba_test_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_json_file, 'w') as f:
        json.dump({"Set": "test", "Actions": test_predictions["Actions"]}, f, indent=4)
    print(f"Predictions saved to '{output_json_file}'")
    
    # Compute performance metrics
    metrics = compute_performance(test_loader.dataset, test_predictions, save_dir="results")
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    test_dataset = MVFoulTestDataset(
        data_dir="/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed",
        json_path="/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed/annotations.json",
        split='test',
        preload=True
    )
    if len(test_dataset) == 0:
        print("Error: No .pt files found in the dataset directory. Please check the data_dir path.")
        exit(1)
    
    try:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate
        )
    except OSError as e:
        print(f"Error initializing DataLoader: {str(e)}")
        print("Possible causes:")
        print("- Temporary directory (/tmp) is full. Check disk space with 'df -h /tmp'.")
        print("- Insufficient memory or swap space. Check with 'free -h'.")
        print("- Too many open files. Check limits with 'ulimit -n'.")
        print("Suggestions:")
        print("- Free up space in /tmp: 'rm -rf /tmp/*' (ensure no critical files are deleted).")
        print("- Set TMPDIR to a directory with more space: 'export TMPDIR=/home/areyesan/tmp'.")
        print("- Increase file descriptor limit: 'ulimit -n 4096'.")
        print("- Run with num_workers=0 (already set).")
        exit(1)

    # Load the trained MultiTaskModel
    model = MultiTaskModelMamba() 

    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model_path = "/kaggle/input/mvfr-v4/pytorch/default/1/best_multitask_mamba_model_epoch5_ba31.2105.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)

    # Supongamos que 'model' es tu modelo de PyTorch
    """
    print("LAYERS: ")
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    """
    
    print("Predicting on Test Split with Mamba Model...")
    metrics = predict_test_split(model, test_loader, device)
