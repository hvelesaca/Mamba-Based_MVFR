import os
import cv2
import torch
import json
import numpy as np

def preprocess_video(data_dir, json_path, output_dir, num_frames=16, center_frame=75):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        metadata = json.load(f)["Actions"]
    action_folders = [d for d in os.listdir(data_dir) if d.startswith("action_")]
    
    for action_id in action_folders:
        action_dir = os.path.join(data_dir, action_id)
        action_num = action_id.replace("action_", "")
        video_files = [f for f in os.listdir(action_dir) if f.endswith(".mp4")]
        video_files.sort()
        clips = []
        
        for video_file in video_files:
            video_path = os.path.join(action_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open {video_path}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Processing {video_path}: Total frames={total_frames}, FPS={fps:.2f}, Resolution={width}x{height}")
            
            # Ensure the video resolution is 224x398
            if width != 224 or height != 398:
                print(f"Warning: {video_path} has resolution {width}x{height}, expected 224x398")
            
            # Extract 16 frames centered around the 75th frame
            start_frame = max(0, center_frame - num_frames // 2)
            end_frame = min(total_frames, start_frame + num_frames)
            frame_indices = list(range(start_frame, end_frame))
            
            if len(frame_indices) < num_frames:
                frame_indices.extend([frame_indices[-1]] * (num_frames - len(frame_indices)))
            elif len(frame_indices) > num_frames:
                frame_indices = frame_indices[:num_frames]
            
            frames = []
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Store as uint8 (0-255 range)
                    frame = torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1)  # [3, 398, 224]
                    frames.append(frame)
                else:
                    frames.append(frames[-1])
            cap.release()
            clips.append(torch.stack(frames))
        
        clips = torch.stack(clips).permute(0, 2, 1, 3, 4)  # [num_views, 3, 16, 398, 224]
        torch.save(clips, os.path.join(output_dir, f"{action_id}.pt"))
        print(f"Saved {action_id}.pt with shape {clips.shape} and dtype {clips.dtype}")

if __name__ == "__main__":
    preprocess_video("dataset/valid", "dataset/valid/annotations.json", "dataset/valid_preprocessed", num_frames=16)
    preprocess_video("dataset/test", "dataset/test/annotations.json", "dataset/test_preprocessed", num_frames=16)
    preprocess_video("dataset/train", "dataset/train/annotations.json", "dataset/train_preprocessed", num_frames=16)

