import glob, numpy as np, torch
import torch.nn as nn
from collections import Counter

class TemporalCNN(nn.Module):
    def __init__(self, d_in, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.net(x.transpose(1,2)).squeeze(-1))

def zscore(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

def fix_dim(X, d_target):
    # X: (T, D) -> (T, d_target)
    D = X.shape[1]
    if D == d_target:
        return X
    if D > d_target:
        return X[:, :d_target]
    pad = np.zeros((X.shape[0], d_target - D), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

def fname_label(p):
    return p.split("/")[-1].split("_")[1].lower()

ckpt = torch.load("word_model.pt", map_location="cpu")
d_in = int(ckpt["d_in"])
num_classes = int(ckpt["num_classes"])

model = TemporalCNN(d_in, num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval()

id_to_word = ckpt.get("id_to_word", {})

paths = glob.glob("clips_npz/*.npz")
correct = 0
total = 0
conf_sum = 0.0
cm = Counter()

for p in paths:
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)        # (T, D_varies)
    X = fix_dim(X, d_in)                 # ✅ match model channels
    X = zscore(X)

    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X).unsqueeze(0)), dim=1)[0].numpy()

    pred_id = int(probs.argmax())
    conf = float(probs[pred_id])

    pred_word = id_to_word.get(pred_id, str(pred_id)) if isinstance(id_to_word, dict) else str(pred_id)
    true_word = fname_label(p)

    cm[(true_word, pred_word)] += 1
    correct += int(pred_word == true_word)
    total += 1
    conf_sum += conf

print("dataset acc:", correct/total if total else 0.0)
print("avg conf:", conf_sum/total if total else 0.0)
print("top confusions:", cm.most_common(10))
print("model d_in:", d_in)
