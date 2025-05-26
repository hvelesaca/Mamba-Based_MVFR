import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model_2_v5 import MultiTaskModelMamba
from tqdm import tqdm
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cv2

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MVFoulDataset(Dataset):
    foul_map = {"No Offence": 0, "Offence Severity 1": 1, "Offence Severity 3": 2, "Offence Severity 5": 3}
    action_map = {
        "Standing Tackling": 0, "Tackling": 1, "Holding": 2, "Pushing": 3,
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
                    self.valid_action_folders.append(folder)
                    self.foul_labels.append(-1)
                    self.action_labels.append(-1)
                    continue
            
            offence = self.metadata[action_id]["Offence"].lower()
            severity_str = self.metadata[action_id]["Severity"]
            action_class = self.metadata[action_id]["Action class"].lower()
            
            normalized_action = self.action_normalization.get(action_class, action_class.title())
            
            if self.filter_data:
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
                foul_label = -1
                action_label = -1
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
    reverse_foul_map = {v: k for k, v in MVFoulDataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in MVFoulDataset.action_map.items()}
    
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
            "confidence_offence": float(foul_probs[foul_pred_idx])
        }
    return predictions

def get_feature_map_heatmap(model, clips, device='cuda'):
    model.eval()
    clips = clips.unsqueeze(1).to(device)  # Add view dimension: [B, V, C, T, H, W], here V=1
    
    # Register hook to capture activations
    activations = []
    def forward_hook(module, input, output):
        activations.append(output.detach())

    if isinstance(model, torch.nn.DataParallel):
        backbone = model.module.backbone
    else:
        backbone = model.backbone
            
    target_layer = backbone.conv_proj
    handle = target_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    with torch.no_grad():
        model(clips)
    
    # Remove hook
    handle.remove()
    
    # Process activations
    activation = activations[0]  # Shape: [B, C_out, T, H, W]
    heatmap = torch.mean(activation, dim=1).squeeze().cpu()  # Average over channels, Shape: [T, H, W]
    heatmap = F.relu(heatmap)  # Keep positive contributions
    heatmap /= torch.max(heatmap) + 1e-8  # Normalize to [0, 1]
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap.numpy(), (image.shape[2], image.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    image = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    image = (image * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0.0)
    return overlay

def visualize_predictions(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    test_predictions = {}
    os.makedirs("outputs", exist_ok=True)
    
    reverse_foul_map = {v: k for k, v in MVFoulDataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in MVFoulDataset.action_map.items()}
    
    with torch.no_grad():
        for batch_clips, foul_labels, action_labels, action_ids in tqdm(test_loader, desc="Generating Predictions and Visualizations", unit="batch"):
            batch_clips = batch_clips.to(device, non_blocking=True)
            foul_logits, action_logits = model(batch_clips)
            
            # Generate predictions
            batch_predictions = generate_predictions_json(action_ids, foul_logits, action_logits)
            test_predictions.update(batch_predictions["Actions"])
            
            # Process each sample in the batch for visualization
            for i, (clips, action_id) in enumerate(zip(batch_clips, action_ids)):
                foul_pred_idx = torch.argmax(foul_logits[i]).item()
                action_pred_idx = torch.argmax(action_logits[i]).item()
                foul_label = reverse_foul_map[foul_pred_idx]
                action_label = reverse_action_map[action_pred_idx]
                
                # Get feature map heatmap for frame 8 (middle of 16 frames)
                frame_idx = 7  # 0-based index for 8th frame
                for view_idx in range(2):  # Two views
                    view_clips = clips[view_idx:view_idx+1]  # Shape: [1, C, T, H, W]
                    heatmap = get_feature_map_heatmap(model, view_clips, device=device)
                    
                    # Extract frame 8 from the view
                    frame = view_clips[0, :, frame_idx]  # Shape: [C, H, W]
                    
                    # Overlay heatmap (same for both foul and action since it's not class-specific)
                    overlay = overlay_heatmap(frame, heatmap[frame_idx])
                    
                    # Create figure with single plot (since heatmap is shared)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.imshow(overlay)
                    ax.axis('off')
                    ax.set_title(f"Foul: {foul_label}\nAction: {action_label}", fontsize=10)
                    
                    # Save image
                    output_path = f"outputs/{action_id}_view{view_idx+1}_featuremap.png"
                    plt.savefig(output_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    print(f"Saved visualization to '{output_path}'")
    
    # Save predictions in SoccerNet format
    output_json = {
        "Set": "test",
        "Actions": {k: test_predictions[k] for k in sorted(test_predictions.keys(), key=int)}
    }
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file = f"results/predictions_multitask_with_featuremap_{timestamp}.json"
    with open(prediction_file, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"\nPredictions saved to '{prediction_file}'")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load test dataset (filtered by default)
    test_dataset = MVFoulDataset("/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed", "/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed/annotations.json", 
                                 split='test', preload=True, filter_data=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, 
                             num_workers=0, pin_memory=True)  # Batch size 1 for individual visualization
    
    # Load the trained MultiTaskModel
    #multitask_model = MultiTaskModel(agr_type='max').to(device)
    #multitask_model.load_state_dict(torch.load("/kaggle/input/mvfr-v4/pytorch/default/1/best_multitask_mamba_model_epoch5_ba31.2105.pth", map_location=device))

    # Load the trained MultiTaskModel
    model = MultiTaskModelMamba() 

    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model_path = "/kaggle/input/mvfr-v5/pytorch/default/1/best_multitask_mamba_model_epoch6_ba33.1056.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    
    print("\nGenerating Predictions and Feature Map Visualizations...")
    visualize_predictions(model, test_loader, device)
