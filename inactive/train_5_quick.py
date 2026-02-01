import os, glob, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "clips_npz"
OUT_CKPT = "word_model.pt"

LABELS = ["hello", "yes", "no", "please", "thanks"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

def clip_to_feat(X: np.ndarray) -> np.ndarray:
    # X: (T, D)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)  # (2D,)

class NPZWords(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i], allow_pickle=True)
        X = d["X"].astype(np.float32)          # (T, 177)
        label = str(d["label"])
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unknown label '{label}' in {self.paths[i]}")
        y = LABEL_TO_ID[label]
        feat = clip_to_feat(X)                # (354,)
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def stratified_split(paths, seed=42, train_frac=0.7, val_frac=0.15):
    rng = random.Random(seed)
    by_label = {l: [] for l in LABELS}

    # collect
    for p in paths:
        d = np.load(p, allow_pickle=True)
        lbl = str(d["label"])
        if lbl in by_label:
            by_label[lbl].append(p)

    # quick label count sanity check
    print("Label counts:")
    for l in LABELS:
        print(f"  {l:7s}: {len(by_label[l])}")

    train, val, test = [], [], []
    for lbl, ps in by_label.items():
        rng.shuffle(ps)
        n = len(ps)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train += ps[:n_train]
        val += ps[n_train:n_train + n_val]
        test += ps[n_train + n_val:]

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not paths:
        raise RuntimeError(f"No .npz files found in {DATA_DIR}/")

    train_paths, val_paths, test_paths = stratified_split(paths)
    print(f"Split sizes: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    train_loader = DataLoader(NPZWords(train_paths), batch_size=32, shuffle=True)
    val_loader   = DataLoader(NPZWords(val_paths),   batch_size=64, shuffle=False)
    test_loader  = DataLoader(NPZWords(test_paths),  batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = 2 * np.load(paths[0], allow_pickle=True)["X"].shape[1]  # 2D
    model = MLP(in_dim=in_dim, num_classes=len(LABELS)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0

    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / max(1, len(train_loader.dataset))
        val_acc = accuracy(model, val_loader, device)

        print(f"ep {epoch:02d} | train loss {train_loss:.4f} | val acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model_state": model.state_dict(), "labels": LABELS, "in_dim": in_dim},
                OUT_CKPT
            )
            print(f"  saved {OUT_CKPT} (best so far)")

        # quick early stop for speed
        if epoch >= 10 and val_acc > 0.95:
            break

    # final test eval
    ckpt = torch.load(OUT_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_acc = accuracy(model, test_loader, device)
    print(f"TEST acc: {test_acc:.3f}")

if __name__ == "__main__":
    main()
