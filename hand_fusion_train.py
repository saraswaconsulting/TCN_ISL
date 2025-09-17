import os, glob, argparse, random, numpy as np, torch, torch.nn as nn, cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from common import load_json, make_class_index
from pose_transformer import PoseTransformerEncoder
from hand_crop_cnn import HandCropEncoder
from common import _import_mediapipe

def sample_frames_with_hands(video_path, num_frames=16):
    mp = _import_mediapipe()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0,T-1), num_frames).astype(int).tolist()
    holistic = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frames = []
    for i in range(T):
        ok, frame = cap.read()
        if not ok: break
        if i in idxs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            # crop both hands if available
            crops = []
            for lm in [res.left_hand_landmarks, res.right_hand_landmarks]:
                if lm is None: continue
                h,w = frame.shape[:2]
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x0,x1 = max(0,min(xs)), min(w-1,max(xs))
                y0,y1 = max(0,min(ys)), min(h-1,max(ys))
                dx, dy = int((x1-x0)*0.25), int((y1-y0)*0.25)
                x0, x1 = max(0, x0-dx), min(w-1, x1+dx)
                y0, y1 = max(0, y0-dy), min(h-1, y1+dy)
                if x1>x0 and y1>y0:
                    crops.append(frame[y0:y1, x0:x1, :])
            frames.append(crops)
    cap.release()
    holistic.close()
    return frames  # list[list[crops]] length num_frames

class DatasetHandFusion(Dataset):
    def __init__(self, features_root, videos_root, class_to_idx, max_len=32, train=True):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.max_len = max_len
        self.train = train
        for cls, idx in class_to_idx.items():
            feat_dir = Path(features_root)/("train" if train else "val")/cls
            if not feat_dir.exists(): continue
            for f in sorted(glob.glob(str(feat_dir/"*.npy"))):
                rel = Path(f).relative_to(Path(features_root)/("train" if train else "val"))
                vid = Path(videos_root)/rel.with_suffix(".mp4")
                if vid.exists():
                    self.samples.append((f, str(vid), idx))

    def __len__(self): return len(self.samples)

    def _pad_or_crop(self, seq):
        T,D = seq.shape
        if T==self.max_len: return seq
        if T>self.max_len:
            start = random.randint(0, T-self.max_len) if self.train else (T-self.max_len)//2
            return seq[start:start+self.max_len]
        pad = np.repeat(seq[-1][None,:], self.max_len-T, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, i):
        feat_path, vid_path, y = self.samples[i]
        seq = np.load(feat_path).astype(np.float32)
        seq = self._pad_or_crop(seq)
        x_pose = torch.from_numpy(seq)                 # (T,150)
        # collect hand crops from uniformly sampled frames
        crops_per_frame = sample_frames_with_hands(vid_path, num_frames=16)
        # flatten all crops; encoder will average
        flat_crops = [c for lst in crops_per_frame for c in lst]
        return x_pose, flat_crops, torch.tensor(y, dtype=torch.long)

class PoseHandFusion(nn.Module):
    def __init__(self, num_classes, pose_dim=150, d_model=256, hand_embed=128, freeze_hand=True):
        super().__init__()
        self.pose = PoseTransformerEncoder(in_dim=pose_dim, d_model=d_model, num_layers=4)
        self.hand = HandCropEncoder(embed_dim=hand_embed, freeze=freeze_hand)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + hand_embed),
            nn.Dropout(0.3),
            nn.Linear(d_model + hand_embed, num_classes)
        )

    def forward(self, x_pose, hand_crops):
        h_pose = self.pose(x_pose).mean(dim=1)         # (B,d)
        # hand_crops is a list of lists per batch -> process per sample
        embeds = []
        for crops in hand_crops:
            z = self.hand(crops)                       # (1,hand_embed)
            embeds.append(z)
        h_hand = torch.cat(embeds, dim=0)              # (B,hand_embed)
        h = torch.cat([h_pose, h_hand], dim=-1)
        return self.head(h)

def main():
    ap = argparse.ArgumentParser("Pose + Hand-crop Fusion Trainer")
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--videos_root", required=True)
    ap.add_argument("--out_dir", default="checkpoints_handfusion")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4)  # smaller batch due to CV2 work
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--no_freeze_hand", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_to_idx = load_json(os.path.join(args.features_root, "class_to_idx.json"))
    num_classes = len(class_to_idx)

    train_ds = DatasetHandFusion(args.features_root, args.videos_root, class_to_idx, max_len=args.max_len, train=True)
    val_ds   = DatasetHandFusion(args.features_root, args.videos_root, class_to_idx, max_len=args.max_len, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: b)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: b)

    model = PoseHandFusion(num_classes=num_classes, freeze_hand=not args.no_freeze_hand).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best = 0.0

    def pack(batch):
        # list of tuples -> tensors + list of crops
        x_pose = torch.stack([x for x,_,_ in batch]).to(device)
        hand_crops = [hc for _,hc,_ in batch]  # list per sample
        y = torch.stack([y for _,_,y in batch]).to(device)
        return x_pose, hand_crops, y

    for epoch in range(1, args.epochs+1):
        model.train()
        tot, acc, n = 0.0, 0.0, 0
        for batch in train_dl:
            x_pose, hand_crops, y = pack(batch)
            opt.zero_grad(set_to_none=True)
            logits = model(x_pose, hand_crops)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            bs = y.size(0); n += bs
            tot += loss.item() * bs
            acc += (logits.argmax(1)==y).float().sum().item()
        tr_loss, tr_acc = tot/max(1,n), acc/max(1,n)

        model.eval()
        tot, acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                x_pose, hand_crops, y = pack(batch)
                logits = model(x_pose, hand_crops)
                loss = crit(logits, y)
                bs = y.size(0); n += bs
                tot += loss.item() * bs
                acc += (logits.argmax(1)==y).float().sum().item()
        va_loss, va_acc = tot/max(1,n), acc/max(1,n)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "class_to_idx": class_to_idx},
                       os.path.join(args.out_dir, "best_handfusion.pt"))
            print(f"  ✔ Saved best hand-fusion (val acc {best:.3f})")

if __name__ == "__main__":
    main()
