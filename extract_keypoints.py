import os, glob, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from common import extract_sequence_from_video, make_class_index, save_json

def main():
    p = argparse.ArgumentParser("Extract pose+hand features to .npy")
    p.add_argument("--data_root", required=True, help="data/ with train/ and val/ folders")
    p.add_argument("--out_root", required=True, help="features/ directory")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    for split in ["train", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Skip: {split_dir} (not found)")
            continue
        class_to_idx = make_class_index(split_dir)
        print(f"[{split}] classes: {class_to_idx}")
        for cls in class_to_idx:
            vids = sorted(glob.glob(str(split_dir / cls / "*.mp4")))
            out_dir = out_root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for vp in tqdm(vids, desc=f"Extract {split}/{cls}"):
                out_path = out_dir / (Path(vp).stem + ".npy")
                if out_path.exists() and not args.overwrite:
                    continue
                seq = extract_sequence_from_video(vp, target_fps=args.fps, max_frames=args.max_frames)
                np.save(out_path, seq)

    # save label map from train split
    train_dir = data_root / "train"
    if train_dir.exists():
        class_to_idx = make_class_index(train_dir)
        save_json(class_to_idx, out_root / "class_to_idx.json")
        print("Saved label map:", out_root / "class_to_idx.json")

if __name__ == "__main__":
    main()
