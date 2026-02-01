# .npz file format collected by record_landmarks_v4_mouthscaled.py
#   X: (T, D) float32
        # --> T --> time step
        # --> D --> feature dimensions
#   ts: (T,) int32
#   label: str
#   speaker: str
#   idxs: (K,) int32  (for debugging / consistency checks)
#   roi: (T, ROI_H, ROI_W) uint8

import os, glob, random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------
# CONSTANTS
# -----------------------------
CLIP_DIR = "clips_npz"
OUT_PATH = "word_model_points_roi.pt"

SEED = 42 # consistent randomization
VAL_FRAC = 0.15 # 15% of all the data gets put into validation

BATCH_SIZE = 16 
EPOCHS = 80 # total number of epochs it will be trained on if it is not stopped early
LR = 3e-4 # learning rate
PATIENCE = 12 # how long the model allows to not improve before pulling the plug early

MAX_T = 90          # pad/trim time to MAX_T frames
DEVICE = "cpu"

USE_ROI_IF_PRESENT = True
ROI_W, ROI_H = 96, 48  # must match recorder if roi exists

# mild augmentation of facial points
NOISE_STD = 0.01 
DROP_FRAMES_PROB = 0.35
DROP_FRAMES_MAX = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# helper functions
# -----------------------------
def split_by_label(files, labels, val_frac=0.15, seed=42):
    """
    because the classes are uneven, we choose to shuffle them by label:
    so every 15% of each label is put into the validation dataset
    """
    rng = random.Random(seed)
    # creates a dictionary with the keys as the labels
    by_lab = defaultdict(list)
    for f, lab in zip(files, labels):
        by_lab[lab].append(f)

    train, val = [], []
    # go through each key value pair
    for lab, fs in by_lab.items():
        rng.shuffle(fs)
        n = len(fs)
        n_val = max(1, int(round(n * val_frac)))
        n_val = min(n_val, n - 1)  # keep at least 1 train
        val.extend(fs[:n_val])
        train.extend(fs[n_val:])
        print(f"{lab:>10}: total={n:4d}  train={n-n_val:4d}  val={n_val:4d}")

    # after placing everything into the respective datasets, shuffle them
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def top_confusions(y_true, y_pred, id_to_label, k=8):
    """
    prints the top k number of the pairings with the greatest errors
    """
    # returns list formatted as ["actual->predicted(number)", ...]
    c = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            c[(t, p)] += 1
    out = []
    for (t, p), n in c.most_common(k):
        out.append(f"{id_to_label[t]}→{id_to_label[p]}({n})")
    return out

def clip_pad_trim(X, T, max_t):
    """
    standardize the features by trimming the long ones and padding them 
    with zeros otherwise
    """
    if T >= max_t:
        return X[:max_t], max_t
    D = X.shape[1]
    out = np.zeros((max_t, D), dtype=np.float32)
    out[:T] = X
    return out, T

def roi_pad_trim(R, T, max_t):
    """
    standardize the features by trimming the long ones and padding them 
    with zeros otherwise
    (we set the max T to be greater than the length of all the usual length of
    the longest clips)
    """
    # R: (T,H,W) uint8
    if T >= max_t:
        return R[:max_t], max_t
    out = np.zeros((max_t, R.shape[1], R.shape[2]), dtype=np.uint8)
    out[:T] = R
    return out, T

# -----------------------------
# daataset
# -----------------------------
class NPZWordDataset(Dataset):
    def __init__(self, files, label_to_id, max_t=90, augment=False, use_roi=True):
        self.files = files
        self.label_to_id = label_to_id
        self.max_t = max_t
        self.augment = augment
        self.use_roi = use_roi

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f, allow_pickle=True)

        X = d["X"].astype(np.float32)  # (T,D)
        T = int(X.shape[0])
        label = str(d["label"])
        y = int(self.label_to_id[label])

        # optional augmentation on point-features
        if self.augment:
            if random.random() < 0.7:
                X = X + np.random.normal(0, NOISE_STD, size=X.shape).astype(np.float32)
            if T > 12 and random.random() < DROP_FRAMES_PROB:
                k = random.randint(1, DROP_FRAMES_MAX)
                drop = np.random.choice(np.arange(1, T - 1), size=k, replace=False)
                keep = np.ones(T, dtype=bool)
                keep[drop] = False
                X = X[keep]
                T = int(X.shape[0])

        # standardize the clips
        X, T_eff = clip_pad_trim(X, T, self.max_t)

        # ROI (optional)
        # if there are ROI data being collected: 
        has_roi = ("roi" in d.files) and self.use_roi
        if has_roi:
            R = d["roi"]  # (Tr,H,W) uint8
            Tr = int(R.shape[0])
            # align lengths conservatively
            T_use = min(T_eff, Tr, self.max_t)
            X = X[:T_use]
            R = R[:T_use]
            # pad both to max_t for batching
            X_pad, T_use = clip_pad_trim(X, T_use, self.max_t)
            R_pad, _ = roi_pad_trim(R, T_use, self.max_t)
            return torch.from_numpy(X_pad), torch.tensor(T_use), torch.from_numpy(R_pad), torch.tensor(y)
        else:
            return torch.from_numpy(X), torch.tensor(T_eff), None, torch.tensor(y)

def collate_fn(batch):
    """
    stacks features, stacks labels, handles optional ROI safely to return batch-ready tensors
    """
    # batch items: (X, T, R_or_None, y)
    Xs, Ts, Rs, ys = [], [], [], []
    any_roi = False
    for X, T, R, y in batch:
        Xs.append(X)
        Ts.append(T)
        ys.append(y)
        if R is not None:
            any_roi = True
        Rs.append(R)

    X = torch.stack(Xs, dim=0)                # (B,MAX_T,D)
    T = torch.stack(Ts, dim=0).long()         # (B,)
    y = torch.stack(ys, dim=0).long()         # (B,)

    if any_roi:
        # if some are missing ROI (shouldn't happen if recorded consistently), fill zeros
        R_fixed = []
        for r in Rs:
            if r is None:
                r = torch.zeros((X.shape[1], ROI_H, ROI_W), dtype=torch.uint8)
            R_fixed.append(r)
        R = torch.stack(R_fixed, dim=0)       # (B,MAX_T,H,W) uint8
    else:
        R = None

    return X, T, R, y

# -----------------------------
# CNN Model for the Mouth ROI
# -----------------------------
class TinyROICNN(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # flattens output
        self.fc = nn.Linear(24, out_dim)

    def forward(self, roi_btchw):
        # roi_btchw: (B,T,1,H,W)
        B, T, C, H, W = roi_btchw.shape
        x = roi_btchw.reshape(B * T, C, H, W)
        x = self.net(x).reshape(B * T, -1)
        x = self.fc(x)
        return x.reshape(B, T, -1)

class AttnPool(nn.Module):
    """
    instead of averaging all time steps equally, the model learns which frames 
    matter more and gives them higher weight.
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, lengths):
        # h: (B,T,H)
        B, T, H = h.shape
        mask = torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B,T)
        scores = self.score(h).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        pooled = (h * w).sum(dim=1)  # (B,H)
        return pooled

# ---------------------------------------------
# GRU Classifier (Gated Reccurent Unit) --> RNN
# ---------------------------------------------
class BiGRUClassifier(nn.Module):
    def __init__(self, x_dim, num_classes, use_roi=False, roi_emb=32, hidden=192):
        super().__init__()
        self.use_roi = use_roi
        self.roi_cnn = TinyROICNN(out_dim=roi_emb) if use_roi else None

        in_dim = x_dim + (roi_emb if use_roi else 0)

        self.gru = nn.GRU(
            in_dim, hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.pool = AttnPool(hidden * 2)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, X, lengths, R=None):
        # B --> batch size
        # T --> time steps
        # D --> feature dimensions (88 landmarks * 2 (for the x and y coords) + openness, area, and velocity)
        # X: (B,T,D), lengths: (B,)
        if self.use_roi:
            # normalize the roi pixels
            r = (R.float() / 255.0).unsqueeze(2)  # (B,T,1,H,W)

            # per-frame standardization (lighting robustness) 
            mu  = r.mean(dim=(2,3,4), keepdim=True)
            std = r.std(dim=(2,3,4), keepdim=True).clamp_min(1e-6)
            r = (r - mu) / std

            # per frame embeddings
            roi_e = self.roi_cnn(r)

            # fuse the point data with the roi
            Z = torch.cat([X, roi_e], dim=2)   # (B,T,D+E)
        else:
            Z = X

        packed = nn.utils.rnn.pack_padded_sequence(
            Z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (B,T,2H)

        # turns the info into one vector per clip (only stores the most relevant data)
        pooled = self.pool(out, lengths)  # (B,2H)
        logits = self.head(pooled)        # (B,C)
        return logits

# -----------------------------
# main
# -----------------------------
def main():
    files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.npz")))
    if not files:
        raise RuntimeError(f"No .npz files found in {CLIP_DIR}")

    labels = []
    dims = []
    has_roi = 0
    idx_signatures = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        labels.append(str(d["label"]))
        dims.append(int(d["X"].shape[1]))
        has_roi += int("roi" in d.files)
        if "idxs" in d.files:
            idx_signatures.append(tuple(d["idxs"].tolist()))
        else:
            idx_signatures.append(None)

    print("Total clips:", len(files))
    print("Label counts:", Counter(labels))
    print("X dims:", Counter(dims))
    print("ROI present in:", has_roi, "files")

    # IMPORTANT: if dims differ, filter to the most common dim
    dim_counter = Counter(dims)
    x_dim = dim_counter.most_common(1)[0][0]
    if len(dim_counter) > 1:
        print("[warn] Multiple feature dims found. Keeping only dim =", x_dim)
        kept = []
        kept_labels = []
        kept_idxsig = []
        for f, lab, d, sig in zip(files, labels, dims, idx_signatures):
            if d == x_dim:
                kept.append(f)
                kept_labels.append(lab)
                kept_idxsig.append(sig)
        files, labels, idx_signatures = kept, kept_labels, kept_idxsig

    # OPTIONAL: warn if idx sets vary a lot
    idx_counter = Counter([s for s in idx_signatures if s is not None])
    if len(idx_counter) > 1:
        most = idx_counter.most_common(1)[0]
        print(f"[warn] Multiple idx signatures detected ({len(idx_counter)}). "
              f"Most common occurs {most[1]} times. "
              f"If accuracy is weird, record using a fixed idx list across clips.")

    uniq = sorted(set(labels))
    label_to_id = {lab:i for i,lab in enumerate(uniq)}
    id_to_label = {i:lab for lab,i in label_to_id.items()}
    num_classes = len(uniq)
    print("Classes:", uniq)

    # split-by-label
    train_files, val_files = split_by_label(files, labels, VAL_FRAC, seed=SEED)
    print("Train clips:", len(train_files), "Val clips:", len(val_files))

    # decide ROI usage
    use_roi = USE_ROI_IF_PRESENT and (has_roi > 0)
    if use_roi:
        print("Using ROI in training.")
    else:
        print("Training WITHOUT ROI.")

    train_ds = NPZWordDataset(train_files, label_to_id, max_t=MAX_T, augment=True, use_roi=use_roi)
    val_ds   = NPZWordDataset(val_files,   label_to_id, max_t=MAX_T, augment=False, use_roi=use_roi)

    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    # --- weighted sampling ---
    train_labels = [str(np.load(f, allow_pickle=True)["label"]) for f in train_files]
    train_counts = Counter(train_labels)

    sample_weights = torch.tensor([1.0 / train_counts[lab] for lab in train_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,      # <-- sampler instead of shuffle
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = BiGRUClassifier(x_dim=x_dim, num_classes=num_classes, use_roi=use_roi, roi_emb=32, hidden=192).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    # --- Class-weighted CE + label smoothing (extra anti-collapse) ---
    # class_weights = torch.tensor(
    #     [1.0 / train_counts[id_to_label[i]] for i in range(num_classes)],
    #     dtype=torch.float32,
    #     device=DEVICE
    # )
    # class_weights = class_weights / class_weights.mean()  # normalize average weight to ~1

    # loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_acc = 0.0
    bad = 0

    for ep in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_ok = 0
        tr_n = 0

        for X, lengths, R, y in train_dl:
            X = X.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)
            if use_roi:
                R = R.to(DEVICE)

            logits = model(X, lengths, R if use_roi else None)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * X.size(0)
            tr_ok += (logits.argmax(dim=1) == y).sum().item()
            tr_n += X.size(0)

        tr_loss /= max(1, tr_n)
        tr_acc = tr_ok / max(1, tr_n)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_ok = 0
        va_n = 0

        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for X, lengths, R, y in val_dl:
                X = X.to(DEVICE)
                lengths = lengths.to(DEVICE)
                y = y.to(DEVICE)
                if use_roi:
                    R = R.to(DEVICE)

                logits = model(X, lengths, R if use_roi else None)
                loss = loss_fn(logits, y)

                va_loss += loss.item() * X.size(0)
                pred = logits.argmax(dim=1)

                va_ok += (pred == y).sum().item()
                va_n += X.size(0)

                y_true_all.extend(y.cpu().tolist())
                y_pred_all.extend(pred.cpu().tolist())

        va_loss /= max(1, va_n)
        va_acc = va_ok / max(1, va_n)

        confs = top_confusions(y_true_all, y_pred_all, id_to_label, k=6)
        conf_str = (" | top confusions: " + ", ".join(confs)) if confs else ""

        print(f"ep {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}{conf_str}")

        if va_acc > best_acc:
            best_acc = va_acc
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "x_dim": x_dim,
                "max_t": MAX_T,
                "use_roi": use_roi,
                "roi_w": ROI_W,
                "roi_h": ROI_H,
                "labels": uniq,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "seed": SEED,
            }, OUT_PATH)
            print(f"  saved {OUT_PATH} (best val acc {best_acc:.3f})")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping. Best val acc: {best_acc:.3f}")
                break

    print("Done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()
