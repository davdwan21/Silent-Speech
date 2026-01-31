# train_words.py
import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CLIP_DIR = "clips_npz"   # where your .npz files are
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-3
MAX_T = 60               # pad/trim to this many frames (adjust if needed)
DEVICE = "cpu"

def pad_or_trim(X, max_t):
    # X: (T, D)
    T, D = X.shape
    if T >= max_t:
        return X[:max_t]
    out = np.zeros((max_t, D), dtype=np.float32)
    out[:T] = X
    return out

class ClipDataset(Dataset):
    def __init__(self, files, label_to_id, max_t=60, augment=True):
        self.files = files
        self.label_to_id = label_to_id
        self.max_t = max_t
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)      # (T, D)
        y = self.label_to_id[str(data["label"])]

        # simple augmentation (helps generalization)
        if self.augment:
            # time jitter: randomly drop or repeat a few frames
            if X.shape[0] > 10 and random.random() < 0.5:
                k = random.randint(1, 3)
                keep = np.ones(X.shape[0], dtype=bool)
                drop_idx = np.random.choice(np.arange(1, X.shape[0]-1), size=k, replace=False)
                keep[drop_idx] = False
                X = X[keep]

            # small Gaussian noise on features
            if random.random() < 0.7:
                X = X + np.random.normal(0, 0.01, size=X.shape).astype(np.float32)

        X = pad_or_trim(X, self.max_t)
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

class GRUWordClassifier(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=20):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)       # (B, T, 2H)
        pooled = out.mean(dim=1)   # simple mean pooling over time
        return self.head(pooled)

def main():
    files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.npz")))
    assert files, f"No .npz found in {CLIP_DIR}"

    # get labels from files
    labels = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        labels.append(str(d["label"]))
    uniq = sorted(set(labels))
    print("Found labels:", uniq)
    label_to_id = {lab:i for i, lab in enumerate(uniq)}
    id_to_label = {i:lab for lab,i in label_to_id.items()}
    num_classes = len(uniq)
    assert num_classes == 20, f"Expected 20 classes, got {num_classes}"

    # split train/val
    rng = list(range(len(files)))
    random.shuffle(rng)
    split = int(0.85 * len(rng))
    train_files = [files[i] for i in rng[:split]]
    val_files   = [files[i] for i in rng[split:]]

    # infer input dim
    sample = np.load(train_files[0], allow_pickle=True)["X"]
    input_dim = sample.shape[1]
    print("Input dim:", input_dim, "MAX_T:", MAX_T)

    train_ds = ClipDataset(train_files, label_to_id, max_t=MAX_T, augment=True)
    val_ds   = ClipDataset(val_files, label_to_id, max_t=MAX_T, augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GRUWordClassifier(input_dim, hidden=128, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        tr_ok = 0
        tr_n = 0
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * X.size(0)
            tr_ok += (logits.argmax(1) == y).sum().item()
            tr_n += X.size(0)

        model.eval()
        va_ok = 0
        va_n = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                va_ok += (logits.argmax(1) == y).sum().item()
                va_n += X.size(0)

        tr_acc = tr_ok / max(1, tr_n)
        va_acc = va_ok / max(1, va_n)
        print(f"ep {ep:02d} | train loss {tr_loss/max(1,tr_n):.4f} | train acc {tr_acc:.3f} | val acc {va_acc:.3f}")

        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(),
                        "label_to_id": label_to_id,
                        "id_to_label": id_to_label,
                        "input_dim": input_dim,
                        "max_t": MAX_T}, "word_model.pt")
            print("  saved word_model.pt (best so far)")

    print("Best val acc:", best)

if __name__ == "__main__":
    main()
