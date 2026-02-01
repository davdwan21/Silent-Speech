import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== CONFIG ==================
MODEL_PATH = "models/face_landmarker.task"

# Point this at your *regular* (non-overfit) checkpoint:
# e.g. "word_model_points_roi.pt" or "word_model_points.pt"
CKPT_PATH  = "word_model_points_roi.pt"

CAM_INDEX  = 1

# must match recorder / training
ROI_W, ROI_H = 96, 48

# Mouth anchors
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

# Distance gating (optional)
MOUTH_W_MIN_PX = 70
MOUTH_W_MAX_PX = 110

DRAW_POINTS = True

# -----------------------------
# FIXED landmark sets (must match your recorder)
# -----------------------------
MOUTH_LOWER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
MOUTH_UPPER = [185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]
CHIN_BOTTOM_ARC = [152,377,400,378,379,394,148,176,149,150,169]
CHEEKS = [
    214, 212, 57, 186, 202, 210, 204, 211, 194, 32,
    83, 201, 208, 18, 200, 199, 313, 421, 428, 396,
    406, 418, 262, 335, 424, 431, 273, 422, 430, 287,
    432, 434, 364, 410, 322, 436, 416
]
FIXED_IDXS = sorted(set(MOUTH_LOWER + MOUTH_UPPER + CHIN_BOTTOM_ARC + CHEEKS))

# ================== MODEL (must match train_model.py) ==================
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
        self.fc = nn.Linear(24, out_dim)

    def forward(self, roi_btchw):
        B, T, C, H, W = roi_btchw.shape
        x = roi_btchw.reshape(B * T, C, H, W)
        x = self.net(x).reshape(B * T, -1)
        x = self.fc(x)
        return x.reshape(B, T, -1)

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, lengths):
        B, T, H = h.shape
        mask = torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * w).sum(dim=1)

class BiGRUClassifier(nn.Module):
    def __init__(self, x_dim, num_classes, use_roi=False, roi_emb=32, hidden=192, gru_layers=2):
        super().__init__()
        self.use_roi = use_roi
        self.roi_cnn = TinyROICNN(out_dim=roi_emb) if use_roi else None
        in_dim = x_dim + (roi_emb if use_roi else 0)

        self.gru = nn.GRU(
            in_dim, hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if gru_layers < 2 else 0.1
        )
        self.pool = AttnPool(hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, X, lengths, R=None):
        if self.use_roi:
            r = (R.float() / 255.0).unsqueeze(2)  # (B,T,1,H,W)
            roi_e = self.roi_cnn(r)
            Z = torch.cat([X, roi_e], dim=2)
        else:
            Z = X

        packed = nn.utils.rnn.pack_padded_sequence(
            Z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        pooled = self.pool(out, lengths)
        return self.head(pooled)

# ================== FEATURE + ROI (match recorder v4_mouthscaled) ==================
def mouth_width_px(face, w, h) -> float:
    L = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], np.float32)
    R = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], np.float32)
    return float(np.linalg.norm(L - R))

def extract_feature(face, w, h, idxs, prev_xy=None):
    xy = np.array([[face[i].x * w, face[i].y * h] for i in idxs], np.float32)
    center = xy.mean(0)

    mouth_w_px = mouth_width_px(face, w, h)
    scale = float(mouth_w_px + 1e-6)

    xy_n = (xy - center) / scale

    if prev_xy is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_n - prev_xy, axis=1)))

    upper = np.array([face[UPPER_INNER].x * w, face[UPPER_INNER].y * h], np.float32)
    lower = np.array([face[LOWER_INNER].x * w, face[LOWER_INNER].y * h], np.float32)
    mouth_open_px = float(np.linalg.norm(upper - lower))
    mouth_aspect = float(mouth_open_px / (mouth_w_px + 1e-6))

    feat = np.concatenate([
        xy_n.reshape(-1),
        np.array([vel, mouth_open_px, mouth_w_px, mouth_aspect], np.float32)
    ])
    return feat, xy_n, center, mouth_w_px

def crop_roi_gray(frame_bgr, center_xy, mouth_w_px):
    h, w = frame_bgr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half_w = 1.2 * mouth_w_px
    half_h = 1.0 * mouth_w_px

    x1 = int(max(0, cx - half_w))
    x2 = int(min(w, cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(h, cy + half_h))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    return roi

def draw_points(frame_bgr, face, idxs, color=(0,255,0)):
    out = frame_bgr
    h, w = out.shape[:2]
    for i in idxs:
        x, y = int(face[i].x * w), int(face[i].y * h)
        cv2.circle(out, (x, y), 1, color, -1)
    return out

# ================== LOAD CKPT ==================
def load_classifier(path):
    ckpt = torch.load(path, map_location="cpu")

    # fields saved by train_model.py
    x_dim = int(ckpt["x_dim"])
    max_t = int(ckpt["max_t"])
    use_roi = bool(ckpt.get("use_roi", False))
    labels = ckpt["labels"]
    id_to_label = ckpt["id_to_label"]

    # Detect GRU layers if present; else default 2
    gru_layers = int(ckpt.get("gru_layers", 2))

    model = BiGRUClassifier(
        x_dim=x_dim,
        num_classes=len(labels),
        use_roi=use_roi,
        roi_emb=32,
        hidden=192,
        gru_layers=gru_layers,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, id_to_label, max_t, use_roi

def topk_from_logits(logits, id_to_label, k=3):
    probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
    top = probs.argsort()[::-1][:k]
    return [(id_to_label[int(i)], float(probs[i])) for i in top]

def distance_gauge_from_mouth_px(mw_px, min_px=70, max_px=110):
    """
    OK        : 70–110 px
    TOO FAR   : < 70 px
    TOO CLOSE : > 110 px
    """
    if mw_px < min_px:
        label = "TOO FAR"
    elif mw_px > max_px:
        label = "TOO CLOSE"
    else:
        label = "OK"

    # Normalize ONLY over the OK range for a stable bar
    t = (mw_px - min_px) / (max_px - min_px + 1e-6)
    t = float(np.clip(t, 0.0, 1.0))

    return t, label

# ================== MAIN ==================
def main():
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"Missing checkpoint: {CKPT_PATH}")

    model, id_to_label, max_t, use_roi = load_classifier(CKPT_PATH)
    print(f"Loaded regular model. use_roi={use_roi} max_t={max_t} classes={len(id_to_label)}")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow("Live Infer", cv2.WINDOW_NORMAL)

    recording = False
    bufX, bufR = [], []
    prev_xy = None
    last_top3 = None

    t0 = time.monotonic()
    with vision.FaceLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = int((time.monotonic() - t0) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp_img, ts)

            out = frame.copy()

            if res.face_landmarks:
                face = res.face_landmarks[0]
                h, w = out.shape[:2]
                mw = mouth_width_px(face, w, h)
                in_range = (MOUTH_W_MIN_PX <= mw <= MOUTH_W_MAX_PX)
                t, dist_label = distance_gauge_from_mouth_px(mw, MOUTH_W_MIN_PX, MOUTH_W_MAX_PX)

                if not recording:
                    cv2.putText(out, f"distance: {dist_label} (mouth_w={mw:.1f}px)",
                                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0,255,0) if dist_label=="OK" else (0,0,255), 2)

                    # draw a little bar
                    bar_x, bar_y, bar_w, bar_h = 20, 215, 200, 12
                    cv2.rectangle(out, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)
                    cv2.rectangle(out, (bar_x, bar_y), (bar_x+int(bar_w*t), bar_y+bar_h), (255,255,255), -1)

                if DRAW_POINTS:
                    draw_points(out, face, FIXED_IDXS, color=(0,255,0))

                # cv2.putText(out, f"mouth_w={mw:.1f}px {'OK' if in_range else 'OUT'}",
                #             (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             (0,255,0) if in_range else (0,0,255), 2)

                if recording and in_range:
                    feat, prev_xy, center, mouth_w = extract_feature(face, w, h, FIXED_IDXS, prev_xy)
                    bufX.append(feat)
                    if use_roi:
                        roi = crop_roi_gray(frame, center, mouth_w)
                        if roi is not None:
                            bufR.append(roi)
                        else:
                            bufR.append(np.zeros((ROI_H, ROI_W), np.uint8))
                else:
                    if recording:
                        prev_xy = None

            status = "REC" if recording else "IDLE"
            cv2.putText(out, f"{status} | r start/stop | q quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            if recording:
                cv2.putText(out, f"frames: {len(bufX)}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if last_top3 is not None:
                y0 = 110
                cv2.putText(out, f"1) {last_top3[0][0]}  {last_top3[0][1]:.2f}",
                            (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if len(last_top3) > 1:
                    cv2.putText(out, f"2) {last_top3[1][0]}  {last_top3[1][1]:.2f}",
                                (20, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if len(last_top3) > 2:
                    cv2.putText(out, f"3) {last_top3[2][0]}  {last_top3[2][1]:.2f}",
                                (20, y0+56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Live Infer", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("r"):
                recording = not recording
                if recording:
                    bufX, bufR = [], []
                    prev_xy = None
                    last_top3 = None
                    print("Recording started...")
                else:
                    print(f"Recording stopped. frames={len(bufX)}. Predicting...")
                    if len(bufX) < 5:
                        print("Too short.")
                        continue

                    feats = np.stack(bufX, axis=0).astype(np.float32)
                    T = min(len(feats), max_t)
                    feats = feats[:T]

                    x = torch.from_numpy(feats).unsqueeze(0)  # (1,T,D)
                    lengths = torch.tensor([T], dtype=torch.long)

                    if use_roi:
                        rois = np.stack(bufR[:T], axis=0).astype(np.uint8)
                        R = torch.from_numpy(rois).unsqueeze(0)  # (1,T,H,W)
                    else:
                        R = None

                    with torch.no_grad():
                        logits = model(x, lengths, R)
                    last_top3 = topk_from_logits(logits, id_to_label, k=3)
                    print("Top3:", last_top3)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()