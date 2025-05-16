import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model_2 import MultiTaskModelMamba
from tqdm import tqdm
import numpy as np
import datetime
import argparse

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MVFoulDataset(Dataset):
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

def generate_groundtruth_json(dataset, task_name):
    groundtruth = {"Actions": {}}
    reverse_foul_map = {v: k for k, v in dataset.foul_map.items()}
    reverse_action_map = {v: k for k, v in dataset.action_map.items()}
    
    for idx in range(len(dataset.action_folders)):
        action_id = dataset.action_folders[idx].replace(".pt", "").replace("action_", "")
        action_class = reverse_action_map.get(dataset.action_labels[idx], "Unknown")
        if task_name == "Foul":
            foul_label = reverse_foul_map.get(dataset.foul_labels[idx], "Unknown")
            if foul_label == "No Offence":
                offence = "No offence"
                severity = ""
            elif foul_label != "Unknown":
                offence = "Offence"
                severity = foul_label.split("Severity ")[1]
            else:
                continue  # Skip if no valid label
            groundtruth["Actions"][action_id] = {
                "Action class": action_class,
                "Offence": offence,
                "Severity": severity
            }
        else:
            if action_class != "Unknown":
                groundtruth["Actions"][action_id] = {
                    "Action class": action_class
                }
    return groundtruth

def custom_evaluate(predictions, groundtruth, task_name):
    if task_name == "Foul":
        num_classes = 4
        class_names = ["No Offence", "Offence Severity 1", "Offence Severity 3", "Offence Severity 5"]
    else:
        num_classes = 8
        class_names = list(MVFoulDataset.action_map.keys())
    
    true_counts = np.zeros(num_classes)
    pred_correct = np.zeros(num_classes)
    pred_counts = np.zeros(num_classes)
    
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
                if pred_offence == "No offence":
                    pred_idx = 0
                elif pred_offence == "Offence":
                    if pred_severity == "1.0":
                        pred_idx = 1
                    elif pred_severity == "3.0":
                        pred_idx = 2
                    elif pred_severity == "5.0":
                        pred_idx = 3
                    else:
                        pred_idx = -1
                else:
                    pred_idx = -1
                if pred_idx == true_idx:
                    pred_correct[true_idx] += 1
                if pred_idx >= 0:
                    pred_counts[pred_idx] += 1
            else:
                pred_action = predictions["Actions"][action_id]["Action class"]
                pred_idx = MVFoulDataset.action_map.get(pred_action, -1)
                if pred_idx == true_idx:
                    pred_correct[true_idx] += 1
                if pred_idx >= 0:
                    pred_counts[pred_idx] += 1
    
    accuracy = sum(pred_correct) / sum(true_counts) if sum(true_counts) > 0 else 0.0
    per_class_acc = {}
    for i, name in enumerate(class_names):
        per_class_acc[name] = pred_correct[i] / true_counts[i] if true_counts[i] > 0 else 0.0
    ba = np.mean(pred_correct / true_counts) if sum(true_counts) > 0 else 0.0
    
    if task_name == "Foul":
        return {
            "accuracy_offence_severity": accuracy * 100,
            "balanced_accuracy_offence_severity": ba * 100,
            "per_class_offence": per_class_acc,
            "true_distribution": true_counts.tolist(),
            "pred_distribution": pred_counts.tolist()
        }
    else:
        return {
            "accuracy_action": accuracy * 100,
            "balanced_accuracy_action": ba * 100,
            "per_class_action": per_class_acc,
            "true_distribution": true_counts.tolist(),
            "pred_distribution": pred_counts.tolist()
        }

def evaluate_multitask_model(model, test_loader, dataset_type, device="cuda"):
    model.eval()

    all_foul_preds, all_action_preds = [], []
    all_foul_labels, all_action_labels = [], []
    all_foul_logits, all_action_logits = [], []
    test_predictions = {}

    with torch.no_grad():
        for batch_clips, foul_labels, action_labels, action_ids in tqdm(test_loader, desc=f"Evaluating {dataset_type.capitalize()} Set"):
            batch_clips = batch_clips.to(device, non_blocking=True)

            foul_logits, action_logits = model(batch_clips)

            all_foul_logits.append(foul_logits.cpu())
            all_action_logits.append(action_logits.cpu())

            foul_preds = torch.argmax(foul_logits, dim=1)
            action_preds = torch.argmax(action_logits, dim=1)

            all_foul_preds.extend(foul_preds.cpu().tolist())
            all_action_preds.extend(action_preds.cpu().tolist())
            all_foul_labels.extend(foul_labels.cpu().tolist())
            all_action_labels.extend(action_labels.cpu().tolist())

            batch_predictions = generate_predictions_json(action_ids, foul_logits, action_logits)
            test_predictions.update(batch_predictions["Actions"])

    # Concatenar logits para métricas adicionales
    all_foul_logits = torch.cat(all_foul_logits)
    all_action_logits = torch.cat(all_action_logits)

    # Guardar predicciones
    output_json = {
        "Set": "test",
        "Actions": {k: test_predictions[k] for k in sorted(test_predictions.keys(), key=int)}
    }
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file = f"results/predictions_multitask_{dataset_type}_{timestamp}.json"
    with open(prediction_file, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"\nPredictions saved to '{prediction_file}'")

    if dataset_type == "filtered":
        # Métricas adicionales
        foul_top2_acc = top_k_accuracy_score(all_foul_labels, all_foul_logits, k=2) * 100
        action_top2_acc = top_k_accuracy_score(all_action_labels, all_action_logits, k=2) * 100

        foul_precision, foul_recall, foul_f1, _ = precision_recall_fscore_support(all_foul_labels, all_foul_preds, average='weighted', zero_division=0)
        action_precision, action_recall, action_f1, _ = precision_recall_fscore_support(all_action_labels, all_action_preds, average='weighted', zero_division=0)

        foul_cm = confusion_matrix(all_foul_labels, all_foul_preds)
        action_cm = confusion_matrix(all_action_labels, all_action_preds)

        # Mostrar resultados
        print("\n📌 Foul Metrics:")
        print(f"Top-1 Accuracy: {np.mean(np.array(all_foul_preds) == np.array(all_foul_labels)) * 100:.4f}%")
        print(f"Top-2 Accuracy: {foul_top2_acc:.4f}%")
        print(f"Precision: {foul_precision:.4f}, Recall: {foul_recall:.4f}, F1-score: {foul_f1:.4f}")
        print("Confusion Matrix (Foul):\n", foul_cm)

        print("\n📌 Action Metrics:")
        print(f"Top-1 Accuracy: {np.mean(np.array(all_action_preds) == np.array(all_action_labels)) * 100:.4f}%")
        print(f"Top-2 Accuracy: {action_top2_acc:.4f}%")
        print(f"Precision: {action_precision:.4f}, Recall: {action_recall:.4f}, F1-score: {action_f1:.4f}")
        print("Confusion Matrix (Action):\n", action_cm)
        
        # Crear carpeta para guardar resultados si no existe
        os.makedirs("results/confusion_matrices", exist_ok=True)
        
        # Matriz de confusión para Foul
        plt.figure(figsize=(8, 6))
        sns.heatmap(foul_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(MVFoulDataset.foul_map.keys()),
                    yticklabels=list(MVFoulDataset.foul_map.keys()))
        plt.title("Foul Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        foul_cm_path = f"results/confusion_matrices/foul_confusion_matrix_{timestamp}.png"
        plt.savefig(foul_cm_path, bbox_inches='tight')
        print(f"✅ Foul confusion matrix saved to '{foul_cm_path}'")
        plt.show()
        plt.close()
        
        # Matriz de confusión para Action
        plt.figure(figsize=(10, 8))
        sns.heatmap(action_cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=list(MVFoulDataset.action_map.keys()),
                    yticklabels=list(MVFoulDataset.action_map.keys()))
        plt.title("Action Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        action_cm_path = f"results/confusion_matrices/action_confusion_matrix_{timestamp}.png"
        plt.savefig(action_cm_path, bbox_inches='tight')
        print(f"✅ Action confusion matrix saved to '{action_cm_path}'")
        plt.show()
        plt.close()

        
        foul_gt_json = generate_groundtruth_json(test_loader.dataset, "Foul")
        action_gt_json = generate_groundtruth_json(test_loader.dataset, "Action")
        
        foul_results = custom_evaluate(output_json, foul_gt_json, "Foul")
        print("\nTest Foul Metrics:")
        print(f"Accuracy: {foul_results['accuracy_offence_severity']:.4f}%, "
              f"Balanced Accuracy: {foul_results['balanced_accuracy_offence_severity']:.4f}%")
        print(f"Per-Class Accuracy: {foul_results['per_class_offence']}")
        print(f"True Distribution: {foul_results['true_distribution']}")
        print(f"Pred Distribution: {foul_results['pred_distribution']}")
        
        action_results = custom_evaluate(output_json, action_gt_json, "Action")
        print("\nTest Action Metrics:")
        print(f"Accuracy: {action_results['accuracy_action']:.4f}%, "
              f"Balanced Accuracy: {action_results['balanced_accuracy_action']:.4f}%")
        print(f"Per-Class Accuracy: {action_results['per_class_action']}")
        print(f"True Distribution: {action_results['true_distribution']}")
        print(f"Pred Distribution: {action_results['pred_distribution']}")
        
def evaluate_multitask_model2(model, test_loader, dataset_type, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    
    all_foul_preds, all_action_preds, all_foul_labels, all_action_labels = [], [], [], []
    test_predictions = {}
    
    with torch.no_grad():
        for batch_clips, foul_labels, action_labels, action_ids in tqdm(test_loader, desc=f"Evaluating {dataset_type.capitalize()} Set", unit="batch"):
            batch_clips = batch_clips.to(device, non_blocking=True)
            
            foul_logits, action_logits = model(batch_clips)
            
            foul_preds = torch.argmax(foul_logits, 1)
            all_foul_preds.extend(foul_preds.cpu().tolist())
            all_foul_labels.extend(foul_labels.cpu().tolist())
            
            action_preds = torch.argmax(action_logits, 1)
            all_action_preds.extend(action_preds.cpu().tolist())
            all_action_labels.extend(action_labels.cpu().tolist())
            
            batch_predictions = generate_predictions_json(action_ids, foul_logits, action_logits)
            test_predictions.update(batch_predictions["Actions"])
    
    # Save predictions in SoccerNet format
    output_json = {
        "Set": "test",
        "Actions": {k: test_predictions[k] for k in sorted(test_predictions.keys(), key=int)}
    }
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_file = f"results/predictions_multitask_{dataset_type}_{timestamp}.json"
    with open(prediction_file, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"\nPredictions saved to '{prediction_file}'")
    
    # If filtered dataset, compute and display metrics
    if dataset_type == "filtered":
        foul_pred_counts = np.bincount(all_foul_preds, minlength=4)
        foul_true_counts = np.bincount(all_foul_labels, minlength=4)
        action_pred_counts = np.bincount(all_action_preds, minlength=8)
        action_true_counts = np.bincount(all_action_labels, minlength=8)
        
        foul_gt_json = generate_groundtruth_json(test_loader.dataset, "Foul")
        action_gt_json = generate_groundtruth_json(test_loader.dataset, "Action")
        
        foul_results = custom_evaluate(output_json, foul_gt_json, "Foul")
        print("\nTest Foul Metrics:")
        print(f"Accuracy: {foul_results['accuracy_offence_severity']:.4f}%, "
              f"Balanced Accuracy: {foul_results['balanced_accuracy_offence_severity']:.4f}%")
        print(f"Per-Class Accuracy: {foul_results['per_class_offence']}")
        print(f"True Distribution: {foul_results['true_distribution']}")
        print(f"Pred Distribution: {foul_results['pred_distribution']}")
        
        action_results = custom_evaluate(output_json, action_gt_json, "Action")
        print("\nTest Action Metrics:")
        print(f"Accuracy: {action_results['accuracy_action']:.4f}%, "
              f"Balanced Accuracy: {action_results['balanced_accuracy_action']:.4f}%")
        print(f"Per-Class Accuracy: {action_results['per_class_action']}")
        print(f"True Distribution: {action_results['true_distribution']}")
        print(f"Pred Distribution: {action_results['pred_distribution']}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate MultiTaskModel on test dataset.")
    parser.add_argument('--dataset_type', type=str, default='filtered', choices=['filtered', 'whole'],
                        help="Type of dataset to evaluate: 'filtered' (with metrics) or 'whole' (predictions only)")
    parser.add_argument('--model_name', type=str, default='', help="Model name to test results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load test dataset based on dataset_type
    filter_data = (args.dataset_type == 'filtered')
    test_dataset = MVFoulDataset("/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed", "/kaggle/input/datasetmvfd/datasetMVFD/test_preprocessed/annotations.json", 
                                 split='test', preload=True, filter_data=filter_data)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate, 
                             num_workers=0, pin_memory=True)
    
    # Load the trained MultiTaskModel
    model = MultiTaskModelMamba() 

    if torch.cuda.device_count() > 1:
        print("Usando", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    model.load_state_dict(torch.load(args.model_name, map_location=device), strict=False)

    print(f"\nEvaluating Trained MultiTask Model on {args.dataset_type.capitalize()} Test Set...")
    evaluate_multitask_model(model, test_loader, args.dataset_type, device)
