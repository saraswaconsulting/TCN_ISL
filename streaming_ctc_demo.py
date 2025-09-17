# Webcam -> MediaPipe -> sliding window -> PoseTransformer -> CTC greedy decode (streaming)
import argparse, collections, time, numpy as np, cv2, torch, unicodedata
from pose_transformer import PoseTransformerEncoder
from common import features_from_frame, _import_mediapipe

BLANK = "<blank>"

def ctc_greedy_ids(log_probs, blank_id=1):
    # log_probs: (T, C) torch
    ids = log_probs.argmax(-1).cpu().numpy().tolist()
    seq, prev = [], None
    for i in ids:
        if i != blank_id and i != prev:
            seq.append(i)
        prev = i
    return seq

class CTCModel(torch.nn.Module):
    def __init__(self, in_dim=150, d_model=256, nhead=8, num_layers=4, num_classes=100):
        super().__init__()
        self.encoder = PoseTransformerEncoder(in_dim=in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.out = torch.nn.Linear(d_model, num_classes)
    def forward(self, x):
        h = self.encoder(x)           # (B,T,d)
        logits = self.out(h)          # (B,T,C)
        return logits

def main():
    ap = argparse.ArgumentParser("Streaming CTC gloss demo")
    ap.add_argument("--checkpoint", required=True, help="checkpoints_ctc/best_ctc.pt")
    ap.add_argument("--window", type=int, default=48, help="frames per sliding window")
    ap.add_argument("--stride", type=int, default=4, help="decode every N frames")
    ap.add_argument("--min_gloss_len", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    itos, stoi = vocab["itos"], vocab["stoi"]

    model = CTCModel(num_classes=len(itos)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mp = _import_mediapipe()
    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
        refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    buf = collections.deque(maxlen=args.window)
    frame_idx = 0
    hyp_text = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            feat = features_from_frame(res)
            buf.append(feat)

            if len(buf) == args.window and frame_idx % args.stride == 0:
                x = torch.from_numpy(np.stack(buf)[None, ...]).to(device)   # (1,T,150)
                with torch.no_grad():
                    logits = model(x)[0]           # (T,C)
                    logp = torch.log_softmax(logits, dim=-1)
                    ids = ctc_greedy_ids(logp, blank_id=stoi[BLANK])
                    toks = [itos[i] for i in ids if i < len(itos)]
                    # simple smoothing: keep last few tokens unique
                    hyp_text = " ".join(toks[-12:]) if len(toks) >= args.min_gloss_len else ""

            # overlay
            cv2.rectangle(frame, (10, 10), (630, 70), (0,0,0), -1)
            cv2.putText(frame, hyp_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Streaming CTC (q to quit)", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

if __name__ == "__main__":
    main()
