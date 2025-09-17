import os, glob, random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------- MediaPipe import (lazy) ----------
def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception as e:
        raise RuntimeError("mediapipe is required. Install with `pip install mediapipe`.") from e

POSE_LM = 33
HAND_LM = 21

def _landmarks_to_xy(landmarks, count_expected):
    if landmarks is None:
        return np.zeros(count_expected*2, dtype=np.float32)
    pts = []
    for lm in landmarks.landmark:
        pts.extend([lm.x, lm.y])
    arr = np.array(pts, dtype=np.float32)
    if arr.size != count_expected*2:
        out = np.zeros(count_expected*2, dtype=np.float32)
        out[:arr.size] = arr
        return out
    return arr

def features_from_frame(results):
    pose_xy  = _landmarks_to_xy(getattr(results, "pose_landmarks", None), POSE_LM)
    lhand_xy = _landmarks_to_xy(getattr(results, "left_hand_landmarks", None), HAND_LM)
    rhand_xy = _landmarks_to_xy(getattr(results, "right_hand_landmarks", None), HAND_LM)
    feat = np.concatenate([pose_xy, lhand_xy, rhand_xy], axis=0)  # 150 dims
    mu, sigma = feat.mean(), feat.std()
    if sigma < 1e-6: sigma = 1.0
    feat = (feat - mu) / sigma
    return feat.astype(np.float32)

def extract_sequence_from_video(video_path, target_fps=15, max_frames=None,
                                detection_conf=0.5, tracking_conf=0.5):
    import cv2
    mp = _import_mediapipe()
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / float(target_fps))))
    seq = []
    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    ) as holistic:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                feat = features_from_frame(res)
                seq.append(feat)
                if max_frames and len(seq) >= max_frames:
                    break
            idx += 1
    cap.release()
    if len(seq) == 0:
        seq = [np.zeros(150, dtype=np.float32)]
    return np.stack(seq, axis=0)  # (T, 150)

# ---------- Dataset ----------
class SeqDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, max_len=32, train=True):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.max_len = max_len
        self.train = train
        for cls, idx in class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): 
                continue
            for p in glob.glob(os.path.join(cls_dir, "*.npy")):
                self.samples.append((p, idx))

    def __len__(self): 
        return len(self.samples)

    def pad_or_crop(self, seq):
        T, D = seq.shape
        if T == self.max_len:
            return seq
        if T > self.max_len:
            # random crop (train) or center crop (val)
            if self.train:
                import random
                start = random.randint(0, T - self.max_len)
            else:
                start = (T - self.max_len) // 2
            return seq[start:start+self.max_len]
        pad = np.repeat(seq[-1][None, :], self.max_len - T, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, i):
        import numpy as np, random
        path, y = self.samples[i]
        seq = np.load(path)
        if self.train and seq.shape[0] > 8:
            if random.random() < 0.2:
                keep = sorted(random.sample(range(seq.shape[0]), k=max(2,int(0.9*seq.shape[0]))))
                seq = seq[keep]
            if random.random() < 0.2:
                seq = seq + np.random.normal(0, 0.02, size=seq.shape).astype(np.float32)
        seq = self.pad_or_crop(seq)
        x = torch.from_numpy(seq.astype(np.float32))  # (T, 150)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# ---------- Model ----------
class GRUClassifier(nn.Module):
    def __init__(self, in_dim=150, hid=256, num_layers=2, num_classes=10, dropout=0.3, bidir=True):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0,
                          bidirectional=bidir)
        out_dim = hid * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)      # (B, T, H*dir)
        feat = out.mean(dim=1)    # mean over time
        logits = self.head(feat)  # (B, C)
        return logits

# ---------- Helpers ----------
def make_class_index(dir_with_class_folders):
    classes = sorted([d for d in os.listdir(dir_with_class_folders) if os.path.isdir(os.path.join(dir_with_class_folders, d))])
    return {c:i for i,c in enumerate(classes)}

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
