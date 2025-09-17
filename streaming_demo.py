# Real-time demo: webcam -> MediaPipe Holistic -> sliding window -> GRU
import argparse, time, collections, numpy as np, cv2, torch
from common import GRUClassifier, features_from_frame, _import_mediapipe

def main():
    p = argparse.ArgumentParser("Streaming ISL demo (webcam)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--window", type=int, default=32, help="frames in sliding window")
    p.add_argument("--stride", type=int, default=4, help="reclassify every N frames")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    
    # Use model configuration from checkpoint if available, otherwise use command line args
    if "args" in ckpt:
        model_args = ckpt["args"]
        hidden = model_args.hidden
        layers = model_args.layers 
        dropout = model_args.dropout
        bidir = True  # Training script uses bidirectional=True
        print(f"Loaded model config from checkpoint: hidden={hidden}, layers={layers}, dropout={dropout}")
    else:
        hidden = args.hidden
        layers = args.layers
        dropout = args.dropout
        bidir = True  # Default to bidirectional
        print(f"Using command line args: hidden={hidden}, layers={layers}, dropout={dropout}")

    model = GRUClassifier(in_dim=150, hid=hidden, num_layers=layers,
                          num_classes=len(class_to_idx), dropout=dropout, bidir=bidir).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mp = _import_mediapipe()
    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
        refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    buf = collections.deque(maxlen=args.window)
    frame_idx = 0
    label_text = "..."
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            feat = features_from_frame(res)   # (150,)
            buf.append(feat)

            if len(buf) == args.window and frame_idx % args.stride == 0:
                x = torch.from_numpy(np.stack(buf)[None, ...]).to(device)  # (1, T, 150)
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred = int(prob.argmax())
                label_text = f"{idx_to_class[pred]} ({prob[pred]:.2f})"

            # draw label
            cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("ISL Streaming Demo (q to quit)", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

if __name__ == "__main__":
    main()
