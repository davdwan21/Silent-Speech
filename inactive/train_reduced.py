# train_reduced.py
# Train on 5 visually distinct words with temporal features

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CLIP_DIR = "clips_npz"
BATCH_SIZE = 16
EPOCHS = 200  # More epochs
LR = 1e-3
USE_MIXUP = False
MIXUP_ALPHA = 0.2
MAX_T = 60
DEVICE = "cpu"

# 5 words with VERY distinct lip movements
SELECTED_WORDS = [
    "hello",   # H open, E wide, LL, O round
    "water",   # W pursed, A open, T, ER
    "thanks",  # TH tongue, A open, NKS
    "please",  # P closed, L, EE wide, Z
    "apple",   # A open, PP closed, L, silent E
]

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blend samples together."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def add_temporal_features(X):
    """Add velocity and acceleration features."""
    if len(X) < 3:
        # Pad with zeros if too short
        velocity = np.zeros_like(X)
        accel = np.zeros_like(X)
    else:
        # Velocity (first derivative)
        velocity = np.zeros_like(X)
        velocity[1:] = X[1:] - X[:-1]

        # Acceleration (second derivative)
        accel = np.zeros_like(X)
        accel[2:] = velocity[2:] - velocity[1:-1]

    # Concatenate: position + velocity + acceleration
    return np.concatenate([X, velocity, accel], axis=1).astype(np.float32)


def pad_or_trim(X, max_t):
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = self.label_to_id[str(data["label"])]

        # Skip temporal features - they hurt performance
        # X = add_temporal_features(X)

        if self.augment:
            # Time warping
            if random.random() < 0.5 and X.shape[0] > 10:
                scale = random.uniform(0.8, 1.2)
                new_len = max(5, int(X.shape[0] * scale))
                indices = np.linspace(0, X.shape[0] - 1, new_len).astype(int)
                X = X[indices]

            # Frame dropping
            if X.shape[0] > 15 and random.random() < 0.3:
                k = random.randint(1, 3)
                keep = sorted(random.sample(range(X.shape[0]), X.shape[0] - k))
                X = X[keep]

            # Noise
            if random.random() < 0.5:
                X = X + np.random.normal(0, 0.015, size=X.shape).astype(np.float32)

            # Scale jitter
            if random.random() < 0.3:
                X = X * random.uniform(0.95, 1.05)

        X = pad_or_trim(X, self.max_t)
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)


def main():
    # Filter files to only selected words
    all_files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.npz")))

    files = []
    for f in all_files:
        label = str(np.load(f, allow_pickle=True)["label"])
        if label in SELECTED_WORDS:
            files.append(f)

    print(f"Using {len(files)} clips from {len(SELECTED_WORDS)} words")

    # Count per word
    from collections import Counter
    labels = [str(np.load(f, allow_pickle=True)["label"]) for f in files]
    print("Distribution:", dict(Counter(labels)))

    label_to_id = {w: i for i, w in enumerate(SELECTED_WORDS)}
    id_to_label = {i: w for w, i in label_to_id.items()}

    # Stratified split - ensure balanced classes
    from collections import defaultdict
    files_by_label = defaultdict(list)
    for f in files:
        label = str(np.load(f, allow_pickle=True)["label"])
        files_by_label[label].append(f)

    train_files, val_files = [], []
    for label, label_files in files_by_label.items():
        random.shuffle(label_files)
        split = int(0.85 * len(label_files))
        train_files.extend(label_files[:split])
        val_files.extend(label_files[split:])

    random.shuffle(train_files)
    random.shuffle(val_files)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Input dim
    sample = np.load(train_files[0], allow_pickle=True)["X"]
    input_dim = sample.shape[1]
    print(f"Input dim: {input_dim}")

    train_ds = ClipDataset(train_files, label_to_id, MAX_T, augment=True)
    val_ds = ClipDataset(val_files, label_to_id, MAX_T, augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GRUClassifier(input_dim, num_classes=len(SELECTED_WORDS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    patience = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            if USE_MIXUP:
                X_mixed, y_a, y_b, lam = mixup_data(X, y, MIXUP_ALPHA)
                logits = model(X_mixed)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(X)
                loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * X.size(0)
            tr_ok += (logits.argmax(1) == y).sum().item()
            tr_n += X.size(0)

        model.eval()
        va_ok, va_n = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                va_ok += (logits.argmax(1) == y).sum().item()
                va_n += X.size(0)

        tr_acc = tr_ok / max(1, tr_n)
        va_acc = va_ok / max(1, va_n)

        scheduler.step(va_acc)
        lr = optimizer.param_groups[0]['lr']

        print(f"ep {ep:02d} | loss {tr_loss/tr_n:.4f} | train {tr_acc:.3f} | val {va_acc:.3f} | lr {lr:.5f}")

        if va_acc > best:
            best = va_acc
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "id_to_label": id_to_label,
                "label_to_id": label_to_id,
                "input_dim": input_dim,
                "max_t": MAX_T,
                "words": SELECTED_WORDS
            }, "word_model_5.pt")
            print("  saved word_model_5.pt (best)")
        else:
            patience += 1
            if patience >= 40:
                print("Early stopping")
                break

    print(f"\nBest validation accuracy: {best:.3f}")
    print(f"Random baseline: {1/len(SELECTED_WORDS):.3f}")


if __name__ == "__main__":
    main()
