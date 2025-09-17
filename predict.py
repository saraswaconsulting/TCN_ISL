import argparse, torch, numpy as np
from common import extract_sequence_from_video, GRUClassifier

def main():
    p = argparse.ArgumentParser("Predict sign from a single video")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
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

    seq = extract_sequence_from_video(args.video, target_fps=args.fps, max_frames=args.max_frames)
    T = seq.shape[0]
    if T >= max_len:
        start = (T - max_len)//2
        seq = seq[start:start+max_len]
    else:
        pad = np.repeat(seq[-1][None,:], max_len-T, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, D)

    model = GRUClassifier(in_dim=150, hid=hidden, num_layers=layers,
                          num_classes=len(class_to_idx), dropout=dropout, bidir=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(prob.argmax())
    print(f"Predicted: {idx_to_class[pred]} (p={prob[pred]:.3f})")
    topk = min(5, len(prob))
    top_indices = np.argsort(-prob)[:topk]
    print("Top-k:")
    for i in top_indices:
        print(f"  {idx_to_class[i]:20s}  {prob[i]:.3f}")

if __name__ == "__main__":
    main()
