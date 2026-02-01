import os, glob, re, random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NPZ_DIR = "clips_npz"
OUT_PT = "word_model.pt"

SEED = 42
BATCH = 64
EPOCHS = 60
LR = 3e-4

# Clip processing
T_TARGET = 32      # set to median clip length (your sample is 32)
MARGIN = 2         # frames around detected "active" region
Q = 0.60           # activity quantile threshold (higher = more aggressive trim)
MIN_KEEP = 6       # if too few active frames, don't trim
USE_DELTAS = True  # highly recommended

# filename: me_<label>_<time>_<idx>.npz
FNAME_RE = re.compile(r"^me_([A-Za-z]+)_\d+_\d+\.npz$")


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def pad_or_trim_time(X: np.ndarray, T: int) -> np.ndarray:
    if X.shape[0] >= T:
        return X[:T]
    pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=X.dtype)
    return np.vstack([X, pad])


def fix_dim(X: np.ndarray, d_target: int) -> np.ndarray:
    D = X.shape[1]
    if D == d_target:
        return X
    if D > d_target:
        return X[:, :d_target]
    pad = np.zeros((X.shape[0], d_target - D), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def zscore_per_clip(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std


def activity_from_X(X: np.ndarray) -> np.ndarray:
    """
    Use openness feature if present (odd D => last column is openness in your data),
    else approximate using y-spread.
    """
    if X.shape[1] % 2 == 1:   # 345 looks like 344 xy + 1 scalar
        return X[:, -1].astype(np.float32)
    y = X[:, 1::2]
    return (y.max(axis=1) - y.min(axis=1)).astype(np.float32)


def trim_clip(X: np.ndarray, T_target: int, margin=MARGIN, q=Q, min_keep=MIN_KEEP) -> np.ndarray:
    a = activity_from_X(X)
    thr = float(np.quantile(a, q))
    active = np.where(a > thr)[0]

    if len(active) < min_keep:
        return pad_or_trim_time(X, T_target)

    lo = max(int(active[0]) - margin, 0)
    hi = min(int(active[-1]) + margin + 1, X.shape[0])
    X2 = X[lo:hi]
    return pad_or_trim_time(X2, T_target)


def add_deltas(X: np.ndarray) -> np.ndarray:
    dX = np.zeros_like(X)
    dX[1:] = X[1:] - X[:-1]
    return np.concatenate([X, dX], axis=1)


class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, num_classes: int, hidden=128, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)     # h: (1,B,H)
        h = self.drop(h[0])    # (B,H)
        return self.head(h)


class NpzWordDataset(Dataset):
    def __init__(self, paths, word_to_id, d_target):
        self.paths = paths
        self.word_to_id = word_to_id
        self.d_target = d_target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        z = np.load(p, allow_pickle=True)
        X = z["X"].astype(np.float32)         # (T, D_var)

        # force fixed feature dim to match model
        X = fix_dim(X, self.d_target)

        # remove silence / dead air
        X = trim_clip(X, T_TARGET)

        # normalize
        X = zscore_per_clip(X)

        # add motion
        if USE_DELTAS:
            X = add_deltas(X)
            X = zscore_per_clip(X)

        # label from filename
        w = os.path.basename(p).split("_")[1].lower()
        y = self.word_to_id[w]

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    set_seed(SEED)

    paths = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No npz files in {NPZ_DIR}")

    # labels from filenames
    words = []
    for p in paths:
        fn = os.path.basename(p)
        parts = fn.split("_")
        if len(parts) >= 4:
            words.append(parts[1].lower())
    word_list = sorted(set(words))
    word_to_id = {w:i for i,w in enumerate(word_list)}
    id_to_word = {i:w for w,i in word_to_id.items()}

    print("Words:", word_list)
    print("Counts:", Counter(words))

    # choose d_target = max D across dataset (matches your earlier approach)
    Ds = []
    for p in paths[:200]:
        z = np.load(p, allow_pickle=True)
        Ds.append(int(z["X"].shape[1]))
    d_target = max(Ds)  # you can also scan all files if you want
    print("Using d_target =", d_target)

    # split
    random.shuffle(paths)
    n = len(paths)
    n_train = int(0.8 * n)
    train_paths = paths[:n_train]
    val_paths = paths[n_train:]

    train_ds = NpzWordDataset(train_paths, word_to_id, d_target)
    val_ds = NpzWordDataset(val_paths, word_to_id, d_target)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    device = get_device()

    d_in = d_target * (2 if USE_DELTAS else 1)
    model = GRUClassifier(d_in=d_in, num_classes=len(word_list)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_correct = tr_total = 0
        tr_loss_sum = 0.0

        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss_sum += float(loss.item()) * y.size(0)
            tr_correct += int((logits.argmax(1) == y).sum().item())
            tr_total += int(y.size(0))

        model.eval()
        va_correct = va_total = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                va_correct += int((logits.argmax(1) == y).sum().item())
                va_total += int(y.size(0))

        tr_acc = tr_correct / max(1, tr_total)
        va_acc = va_correct / max(1, va_total)
        tr_loss = tr_loss_sum / max(1, tr_total)

        print(f"ep {ep:03d} | loss {tr_loss:.4f} | train {tr_acc:.3f} | val {va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "d_in": d_in,
                "num_classes": len(word_list),
                "word_to_id": word_to_id,
                "id_to_word": id_to_word,
                "t_target": T_TARGET,
                "d_target": d_target,
                "use_deltas": USE_DELTAS,
                "trim": {"q": Q, "margin": MARGIN, "min_keep": MIN_KEEP},
            }
            torch.save(ckpt, OUT_PT)
            print(f"  saved {OUT_PT} (best val {best_val:.3f})")

    print("best val:", best_val)


if __name__ == "__main__":
    main()
