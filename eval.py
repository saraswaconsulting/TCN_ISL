import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from common import SeqDataset, GRUClassifier, load_json

def main():
    p = argparse.ArgumentParser("Evaluate GRU classifier")
    p.add_argument("--features_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--out_dir", default="reports")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    
    # Use model configuration from checkpoint if available
    if "args" in ckpt:
        model_args = ckpt["args"]
        hidden = model_args.hidden
        layers = model_args.layers 
        dropout = model_args.dropout
        max_len = model_args.max_len
        print(f"Loaded model config from checkpoint: hidden={hidden}, layers={layers}, dropout={dropout}, max_len={max_len}")
    else:
        hidden = args.hidden
        layers = args.layers
        dropout = args.dropout
        max_len = args.max_len
        print(f"Using command line args: hidden={hidden}, layers={layers}, dropout={dropout}, max_len={max_len}")

    val_ds = SeqDataset(os.path.join(args.features_root, "val"), class_to_idx, max_len=max_len, train=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = GRUClassifier(in_dim=150, hid=hidden, num_layers=layers,
                          num_classes=len(class_to_idx), dropout=dropout, bidir=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(val_dl, desc="Eval"):
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    # Reports
    os.makedirs(args.out_dir, exist_ok=True)
    report_txt = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(len(idx_to_class))])
    print(report_txt)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    print("Saved reports to", args.out_dir)

if __name__ == "__main__":
    main()
