import os, glob, random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================= CONFIG =================
CLIP_DIR = "clips_npz"
OUT_PATH = "ctc_word_model_roi.pt"

SEED = 42
VAL_FRAC = 0.15
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
PATIENCE = 6

MAX_T = 80
DEVICE = "cpu"

# ROI
ROI_W, ROI_H = 96, 48
ROI_EMB = 32

# Length penalty (keep small!)
LEN_LAMBDA = 0.02   # set 0.0 to disable

# ================= VOCAB =================
VOCAB = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz")
BLANK_ID = 0
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CHAR = {i: c for c, i in CHAR2ID.items()}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= HELPERS =================
def normalize_label(word: str) -> str:
    return "".join(ch for ch in word.lower() if "a" <= ch <= "z")

def encode_text(text: str):
    return [CHAR2ID[ch] for ch in text]

def trim_silence_pair_np(X, R, open_idx=-3, thresh=0.05, pad=2):
    if len(X) == 0:
        return X, R
    o = X[:, open_idx]
    active = np.where(o > thresh)[0]
    if len(active) == 0:
        return X, R
    s = max(0, active[0] - pad)
    e = min(len(X), active[-1] + pad + 1)
    return X[s:e], R[s:e]

# ================= DATASET =================
class NPZCTCDataset(Dataset):
    def __init__(self, files, label_to_text, augment=False):
        self.files = files
        self.label_to_text = label_to_text
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)

        X = d["X"].astype(np.float32)          # (T,D)
        R = d["roi"].astype(np.uint8)          # (T,H,W)
        label = str(d["label"])
        text = self.label_to_text[label]

        X, R = trim_silence_pair_np(X, R)

        if self.augment:
            if random.random() < 0.6:
                X += np.random.normal(0, 0.01, X.shape).astype(np.float32)

        T = min(len(X), MAX_T)
        X = X[:T]
        R = R[:T]

        y = np.array(encode_text(text), dtype=np.int32)
        return (
            torch.from_numpy(X),
            torch.from_numpy(R),
            torch.tensor(T, dtype=torch.long),
            torch.from_numpy(y),
            torch.tensor(len(y), dtype=torch.long),
            label,
        )

def collate_ctc(batch):
    Ts = [b[2].item() for b in batch]
    maxT = max(Ts)
    B = len(batch)
    D = batch[0][0].shape[1]

    Xpad = torch.zeros((B, maxT, D))
    Rpad = torch.zeros((B, maxT, 1, ROI_H, ROI_W))
    lengths = torch.tensor(Ts, dtype=torch.long)

    ys, y_lens, labels = [], [], []
    for i, (X, R, T, y, L, lab) in enumerate(batch):
        Xpad[i, :T] = X
        Rpad[i, :T, 0] = torch.from_numpy(R.astype(np.float32) / 255.0)
        ys.append(y)
        y_lens.append(L)
        labels.append(lab)

    return (
        Xpad,
        Rpad,
        lengths,
        torch.cat(ys),
        torch.stack(y_lens),
        labels,
    )

# ================= MODEL =================
class TinyROICNN(nn.Module):
    def __init__(self, out_dim=ROI_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(24, out_dim)

    def forward(self, r):
        B, T, C, H, W = r.shape
        x = r.reshape(B*T, C, H, W)
        x = self.net(x).reshape(B*T, -1)
        return self.fc(x).reshape(B, T, -1)

class BiGRUCTCWithROI(nn.Module):
    def __init__(self, x_dim, hidden=192, num_classes=len(VOCAB)):
        super().__init__()
        self.roi = TinyROICNN()
        self.gru = nn.GRU(
            x_dim + ROI_EMB,
            hidden,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.proj = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, r, lengths):
        r_emb = self.roi(r)
        z = torch.cat([x, r_emb], dim=2)

        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return F.log_softmax(self.proj(out), dim=2).transpose(0, 1)

# ================= CTC WORD SCORING =================
def ctc_word_logprob(log_probs_tc, word_ids):
    ext = [BLANK_ID]
    for cid in word_ids:
        ext += [cid, BLANK_ID]

    S = len(ext)
    T, _ = log_probs_tc.shape
    alpha = torch.full((S,), -1e9)

    alpha[0] = log_probs_tc[0, BLANK_ID]
    if S > 1:
        alpha[1] = log_probs_tc[0, ext[1]]

    for t in range(1, T):
        prev = alpha.clone()
        for s in range(S):
            cand = [prev[s]]
            if s > 0: cand.append(prev[s-1])
            if s > 1 and ext[s] != BLANK_ID and ext[s] != ext[s-2]:
                cand.append(prev[s-2])
            alpha[s] = torch.logsumexp(torch.stack(cand), 0) + log_probs_tc[t, ext[s]]

    return torch.logsumexp(alpha[-2:], 0)

# ================= TRAIN =================
def main():
    files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.npz")))
    labels = [str(np.load(f, allow_pickle=True)["label"]) for f in files]
    uniq = sorted(set(labels))

    label_to_text = {l: normalize_label(l) for l in uniq}
    dict_words = [(l, encode_text(label_to_text[l])) for l in uniq]

    sample = np.load(files[0], allow_pickle=True)
    x_dim = sample["X"].shape[1]

    by_lab = defaultdict(list)
    for f, l in zip(files, labels):
        by_lab[l].append(f)

    train_files, val_files = [], []
    for l, fs in by_lab.items():
        random.shuffle(fs)
        n_val = max(1, int(len(fs) * VAL_FRAC))
        val_files += fs[:n_val]
        train_files += fs[n_val:]

    train_ds = NPZCTCDataset(train_files, label_to_text, augment=True)
    val_ds = NPZCTCDataset(val_files, label_to_text)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_ctc)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_ctc)

    model = BiGRUCTCWithROI(x_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    best, bad = 0.0, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for X, R, L, y, yL, _ in train_dl:
            lp = model(X, R, L)
            loss = ctc(lp, y, L, yL)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        ok, tot = 0, 0
        with torch.no_grad():
            for X, R, L, _, _, labs in val_dl:
                lp = model(X, R, L)
                for b, lab in enumerate(labs):
                    T = L[b]
                    lp_tc = lp[:T, b]
                    best_lab, best_score = None, -1e18
                    for cand, ids in dict_words:
                        s = ctc_word_logprob(lp_tc, ids)
                        if LEN_LAMBDA > 0:
                            s -= LEN_LAMBDA * abs(T - len(ids)*5)
                        if s > best_score:
                            best_score, best_lab = s, cand
                    ok += (best_lab == lab)
                    tot += 1

        acc = ok / tot
        print(f"ep {ep:03d} | val acc {acc:.3f}")

        if acc > best:
            best, bad = acc, 0
            torch.save({
                "model": model.state_dict(),
                "x_dim": x_dim,
                "max_t": MAX_T,
                "vocab": VOCAB,
                "blank_id": BLANK_ID,
                "label_to_text": label_to_text,
                "uniq_labels": uniq,
            }, OUT_PATH)
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    print("Best val acc:", best)

if __name__ == "__main__":
    main()
