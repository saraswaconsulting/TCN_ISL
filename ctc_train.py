import os, argparse, json, math, unicodedata
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pose_transformer import PoseTransformerEncoder

BLANK_TOKEN = "<blank>"
PAD_TOKEN   = "<pad>"

def normalize_token(t: str) -> str:
    t = unicodedata.normalize("NFKC", t.strip())
    return t

class ManifestDataset(Dataset):
    def __init__(self, tsv_path: str, vocab: dict=None, max_len: int=48, build_vocab: bool=False):
        self.samples = []   # list of (path, tokens list)
        self.max_len = max_len
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                path, gloss_str = line.split("\t", 1)
                tokens = [normalize_token(t) for t in gloss_str.split() if t.strip()]
                self.samples.append((path, tokens))
        if build_vocab:
            toks = set()
            for _, ts in self.samples: toks.update(ts)
            itos = [PAD_TOKEN, BLANK_TOKEN] + sorted(toks)
            stoi = {t:i for i,t in enumerate(itos)}
            self.vocab = {"itos": itos, "stoi": stoi}
        else:
            assert vocab is not None, "vocab must be provided when build_vocab=False"
            self.vocab = vocab

    def __len__(self): return len(self.samples)

    def _pad_or_crop(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape
        if T == self.max_len:
            return seq
        if T > self.max_len:
            start = (T - self.max_len)//2
            return seq[start:start+self.max_len]
        pad = np.repeat(seq[-1][None,:], self.max_len-T, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, idx: int):
        path, tokens = self.samples[idx]
        seq = np.load(path).astype(np.float32)   # (T, 150)
        T = seq.shape[0]
        x_len = min(T, self.max_len)
        x = self._pad_or_crop(seq)               # (max_len, 150)

        # targets -> ids (no pad or blank inside target sequence)
        ids = [self.vocab["stoi"][t] for t in tokens if t in self.vocab["stoi"]]
        y = torch.tensor(ids, dtype=torch.long)
        return torch.from_numpy(x), x_len, y, len(ids)

def ctc_greedy_decode(log_probs, input_lengths, blank_id=1):
    # log_probs: (B, T, C)
    preds = log_probs.argmax(-1)  # (B, T)
    out = []
    for b in range(preds.size(0)):
        prev = None
        seq = []
        for t in range(input_lengths[b]):
            p = preds[b, t].item()
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        out.append(seq)
    return out

def compute_wer(ref: List[str], hyp: List[str]) -> float:
    # Simple WER via DP (Levenshtein)
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / max(1, n)

class CTCModel(nn.Module):
    def __init__(self, in_dim=150, d_model=256, nhead=8, num_layers=4, num_classes=100, dropout=0.1):
        super().__init__()
        self.encoder = PoseTransformerEncoder(in_dim=in_dim, d_model=d_model, nhead=nhead,
                                              num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(d_model, num_classes)  # includes blank

    def forward(self, x):
        # x: (B, T, in_dim)
        h = self.encoder(x)         # (B, T, d_model)
        logits = self.out(h)        # (B, T, C)
        return logits

def main():
    ap = argparse.ArgumentParser("CTC gloss recognizer (manifest-based)")
    ap.add_argument("--train_tsv", required=True, help="lines: <path_to_npy>\t<gloss tokens separated by space>")
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--out_dir", default="checkpoints_ctc")
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    # Build vocab from train manifest
    train_ds = ManifestDataset(args.train_tsv, vocab=None, max_len=args.max_len, build_vocab=True)
    vocab = train_ds.vocab
    val_ds   = ManifestDataset(args.val_tsv, vocab=vocab, max_len=args.max_len, build_vocab=False)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=None)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(vocab["itos"])
    model = CTCModel(in_dim=150, d_model=256, nhead=8, num_layers=4, num_classes=num_classes).to(device)

    ctc_loss = nn.CTCLoss(blank=vocab["stoi"]["<blank>"], zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def pack_batch(batch):
        xs, xlens, ys, ylens = zip(*batch)
        xs = torch.stack(xs).to(device)           # (B, T, 150)
        xlens = torch.tensor(xlens, dtype=torch.long).to(device)
        # targets need to be concatenated for CTC
        ycat = torch.cat(ys).to(device)
        ylens = torch.tensor(ylens, dtype=torch.long).to(device)
        return xs, xlens, ycat, ylens

    best_wer = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss, n = 0.0, 0
        for batch in train_dl:
            xs, xlens, ycat, ylens = pack_batch(batch)
            logits = model(xs)                    # (B, T, C)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0,1)  # (T, B, C) for CTC
            loss = ctc_loss(log_probs, ycat, xlens, ylens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = xs.size(0)
            tot_loss += loss.item() * bs
            n += bs
        tr_loss = tot_loss / max(1,n)

        # validation WER
        model.eval()
        wers, m = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                xs, xlens, ycat, ylens = pack_batch(batch)
                logits = model(xs)
                log_probs = torch.log_softmax(logits, dim=-1)
                # greedy decode
                hyp_ids = ctc_greedy_decode(log_probs, xlens, blank_id=vocab["stoi"]["<blank>"])
                # reconstruct references per sample
                offset = 0
                refs = []
                for L in ylens.tolist():
                    refs.append(ycat[offset:offset+L].tolist())
                    offset += L
                for ref_ids, hyp in zip(refs, hyp_ids):
                    ref_toks = [vocab["itos"][i] for i in ref_ids]
                    hyp_toks = [vocab["itos"][i] for i in hyp]
                    wers += compute_wer(ref_toks, hyp_toks)
                    m += 1
        val_wer = wers / max(1,m)
        print(f"Epoch {epoch:02d} | train CTC loss {tr_loss:.3f} | val WER {val_wer:.3f}")

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save({"model": model.state_dict(), "vocab": vocab}, os.path.join(args.out_dir, "best_ctc.pt"))
            print(f"  ✔ Saved best CTC (WER {best_wer:.3f})")

if __name__ == "__main__":
    main()
