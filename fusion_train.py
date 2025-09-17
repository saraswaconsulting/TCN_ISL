import os, glob, argparse, random, cv2, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from pose_transformer import PoseTransformerEncoder
from common import make_class_index, load_json

def sample_video_clip(path, num_frames=16):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, T-1), num_frames).astype(int).tolist()
    frames = []
    for i in range(T):
        ok, frame = cap.read()
        if not ok: break
        if i in idxs:
            frames.append(frame)
    cap.release()
    # If fewer frames than requested, pad by repeating last
    while len(frames) < num_frames:
        frames.append(frames[-1])
    # Preprocess: resize to 112x112, BGR->RGB, to tensor (C,T,H,W)
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[:num_frames]]
    frames = [cv2.resize(f, (112,112), interpolation=cv2.INTER_LINEAR) for f in frames]
    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, C)
    arr = arr.transpose(3,0,1,2)  # (C, T, H, W)
    return torch.from_numpy(arr)

class FusionDataset(Dataset):
    def __init__(self, features_root, videos_root, class_to_idx, max_len=32, train=True):
        self.samples = []
        self.class_to_idx = class_to_idx
        self.max_len = max_len
        self.train = train
        # Expect same relative structure between features and videos
        for cls, idx in class_to_idx.items():
            feat_dir = Path(features_root)/"train"/cls if train else Path(features_root)/"val"/cls
            if not feat_dir.exists(): 
                continue
            for f in sorted(glob.glob(str(feat_dir/"*.npy"))):
                rel = Path(f).relative_to(Path(features_root)/("train" if train else "val"))
                vid = Path(videos_root)/rel.with_suffix(".mp4")
                if not vid.exists():
                    # try same name different ext
                    alt = list(Path(vid).parent.glob(Path(vid).stem + ".*"))
                    if alt:
                        vid = alt[0]
                    else:
                        continue
                self.samples.append((f, str(vid), idx))

    def __len__(self): return len(self.samples)

    def _pad_or_crop(self, seq):
        T, D = seq.shape
        if T == self.max_len: return seq
        if T > self.max_len:
            if self.train:
                start = random.randint(0, T - self.max_len)
            else:
                start = (T - self.max_len)//2
            return seq[start:start+self.max_len]
        pad = np.repeat(seq[-1][None,:], self.max_len-T, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, i):
        feat_path, vid_path, y = self.samples[i]
        seq = np.load(feat_path).astype(np.float32)
        seq = self._pad_or_crop(seq)
        x_pose = torch.from_numpy(seq)                 # (T,150)
        x_rgb  = sample_video_clip(vid_path, 16)       # (C,T,H,W)
        return x_pose, x_rgb, torch.tensor(y, dtype=torch.long)

class TwoStreamFusion(nn.Module):
    def __init__(self, num_classes, pose_dim=150, pose_d_model=256, pose_layers=4, rgb_feat_dim=512, freeze_rgb=True):
        super().__init__()
        self.pose_enc = PoseTransformerEncoder(in_dim=pose_dim, d_model=pose_d_model, num_layers=pose_layers)
        weights = R2Plus1D_18_Weights.DEFAULT
        self.rgb = r2plus1d_18(weights=weights)
        # remove classifier head -> use features before final FC
        self.rgb.fc = nn.Identity()
        if freeze_rgb:
            for p in self.rgb.parameters():
                p.requires_grad=False
        fuse_dim = pose_d_model + 512
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(0.3),
            nn.Linear(fuse_dim, num_classes)
        )

    def forward(self, x_pose, x_rgb):
        # x_pose: (B, T, 150), x_rgb: (B, 3, 16, 112, 112)
        h_pose = self.pose_enc(x_pose)          # (B, T, d)
        h_pose = h_pose.mean(dim=1)             # (B, d)
        h_rgb  = self.rgb(x_rgb)                # (B, 512)
        h = torch.cat([h_pose, h_rgb], dim=-1)  # (B, d+512)
        return self.head(h)

def main():
    ap = argparse.ArgumentParser("Two-Stream Fusion Trainer")
    ap.add_argument("--features_root", required=True)
    ap.add_argument("--videos_root", required=True)
    ap.add_argument("--out_dir", default="checkpoints_fusion")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--freeze_rgb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_to_idx = load_json(os.path.join(args.features_root, "class_to_idx.json"))
    num_classes = len(class_to_idx)

    train_ds = FusionDataset(args.features_root, args.videos_root, class_to_idx, max_len=args.max_len, train=True)
    val_ds   = FusionDataset(args.features_root, args.videos_root, class_to_idx, max_len=args.max_len, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = TwoStreamFusion(num_classes=num_classes, freeze_rgb=args.freeze_rgb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        tot, acc_sum, n = 0.0, 0.0, 0
        for x_pose, x_rgb, y in train_dl:
            x_pose, x_rgb, y = x_pose.to(device), x_rgb.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x_pose, x_rgb)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            bs = y.size(0)
            tot += loss.item() * bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += bs
        tr_loss = tot/max(1,n)
        tr_acc = acc_sum/max(1,n)

        model.eval()
        tot, acc_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x_pose, x_rgb, y in val_dl:
                x_pose, x_rgb, y = x_pose.to(device), x_rgb.to(device), y.to(device)
                logits = model(x_pose, x_rgb)
                loss = crit(logits, y)
                bs = y.size(0)
                tot += loss.item() * bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += bs
        va_loss = tot/max(1,n)
        va_acc = acc_sum/max(1,n)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "class_to_idx": class_to_idx},
                       os.path.join(args.out_dir, "best_fusion.pt"))
            print(f"  ✔ Saved best fusion (val acc {best:.3f})")

if __name__ == "__main__":
    main()
