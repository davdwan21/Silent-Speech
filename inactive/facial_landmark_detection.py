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

MODEL_PATH = "models/face_landmarker.task"
CTC_MODEL_PATH = "ctc_word_model_roi.pt"
CAM_INDEX = 1

# ROI must match what you recorded/trained with
ROI_W, ROI_H = 96, 48

SPEAKER = "me"
SAVE_ROI = True
ROI_W, ROI_H = 96, 48
DRAW_POINTS = True

LOWER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]
UPPER_LIPS = [
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
    311, 312, 13, 82, 81, 42, 183, 78
]

LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))
DEVICE = "cpu"

# ---------------- ROI model (must match training) ----------------
class TinyROICNN(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(24, out_dim)

    def forward(self, roi_btchw):
        B, T, C, H, W = roi_btchw.shape
        x = roi_btchw.reshape(B * T, C, H, W)
        x = self.net(x).reshape(B * T, -1)
        x = self.fc(x)
        return x.reshape(B, T, -1)

class BiGRUCTCWithROI(nn.Module):
    def __init__(self, x_dim, roi_emb=32, hidden=192, num_classes=27):
        super().__init__()
        self.roi_cnn = TinyROICNN(out_dim=roi_emb)
        self.gru = nn.GRU(
            x_dim + roi_emb, hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.proj = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, roi, lengths):
        # x: (B,T,D), roi: (B,T,1,H,W)
        roi_e = self.roi_cnn(roi)              # (B,T,E)
        z = torch.cat([x, roi_e], dim=2)       # (B,T,D+E)

        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (B,T,2H)
        logits = self.proj(out)                # (B,T,C)
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)  # (T,B,C)  <-- like training
        return log_probs

# ---------------- feature + ROI extraction (match recorder) ----------------
def mouth_openness(face, w, h) -> float:
    def p(i):
        lm = face[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    open_dist = np.linalg.norm(p(UPPER_INNER) - p(LOWER_INNER))
    width = np.linalg.norm(p(LEFT_CORNER) - p(RIGHT_CORNER)) + 1e-6
    return float(open_dist / width)

def polygon_area_xy(pts_xy: np.ndarray) -> float:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )

def extract_mouth_features(face, w, h, prev_xy_norm=None):
    xy = np.array([[face[i].x * w, face[i].y * h] for i in MOUTH_SET], dtype=np.float32)
    left = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], dtype=np.float32)
    right = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], dtype=np.float32)
    mouth_w = float(np.linalg.norm(left - right) + 1e-6)

    center = xy.mean(axis=0)
    xy_norm = (xy - center) / mouth_w

    openness = mouth_openness(face, w, h)

    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.array([[face[i].x * w, face[i].y * h] for i in loop_idx], dtype=np.float32)
    loop_xy_norm = (loop_xy - center) / mouth_w
    area = polygon_area_xy(loop_xy_norm)

    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate([xy_norm.reshape(-1), np.array([openness, area, vel], dtype=np.float32)], axis=0)
    return feat, xy_norm, center, mouth_w

def crop_mouth_roi_gray(frame_bgr, center_xy, mouth_width_px):
    h, w = frame_bgr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    half_w = 1.2 * mouth_width_px
    half_h = 0.8 * mouth_width_px

    x1 = int(max(0, cx - half_w))
    x2 = int(min(w, cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(h, cy + half_h))

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    return roi  # uint8 (H,W)

def trim_silence_pair_np(feats, rois, open_idx=-3, thresh=0.05, pad=2):
    # feats: (T,D), rois: (T,H,W)
    if feats.shape[0] == 0:
        return feats, rois
    o = feats[:, open_idx]
    active = np.where(o > thresh)[0]
    if len(active) == 0:
        return feats, rois
    s = max(0, int(active[0]) - pad)
    e = min(len(feats), int(active[-1]) + pad + 1)
    return feats[s:e], rois[s:e]

# ---------------- CTC dictionary scoring ----------------
def ctc_word_logprob(log_probs_tc, word_ids, blank=0):
    ext = [blank]
    for cid in word_ids:
        ext += [cid, blank]
    S = len(ext)

    T, C = log_probs_tc.shape
    neg_inf = -1e9
    alpha = torch.full((S,), neg_inf, device=log_probs_tc.device)

    alpha[0] = log_probs_tc[0, blank]
    if S > 1:
        alpha[1] = log_probs_tc[0, ext[1]]

    for t in range(1, T):
        prev = alpha.clone()
        for s in range(S):
            candidates = [prev[s]]
            if s - 1 >= 0:
                candidates.append(prev[s - 1])
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                candidates.append(prev[s - 2])
            alpha[s] = torch.logsumexp(torch.stack(candidates), dim=0) + log_probs_tc[t, ext[s]]

    if S == 1:
        return alpha[0]
    return torch.logsumexp(torch.stack([alpha[S - 1], alpha[S - 2]]), dim=0)

def load_ctc_model_roi(path):
    ckpt = torch.load(path, map_location="cpu")

    x_dim = int(ckpt.get("x_dim", ckpt.get("input_dim")))
    max_t = int(ckpt["max_t"])
    vocab = ckpt["vocab"]
    blank_id = int(ckpt["blank_id"])
    label_to_text = ckpt["label_to_text"]
    uniq_labels = ckpt["uniq_labels"]

    # NEW: length prior (may be missing)
    exp_len = ckpt.get("exp_len", {})          # label -> mean length
    len_lambda = float(ckpt.get("len_lambda", 0.0))

    char2id = {c: i for i, c in enumerate(vocab)}
    dict_words = []
    for lab in uniq_labels:
        txt = label_to_text[lab]
        ids = [char2id[ch] for ch in txt]
        dict_words.append((lab, ids))

    model = BiGRUCTCWithROI(x_dim=x_dim, roi_emb=32, hidden=192, num_classes=len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, dict_words, max_t, blank_id, vocab, exp_len, len_lambda

def predict_word_ctc_roi(model, dict_words, blank_id, feats, rois, max_t,
                         exp_len=None, len_lambda=0.0):
    if feats.shape[0] < 5:
        return None

    T = min(feats.shape[0], rois.shape[0], max_t)
    feats = feats[:T]
    rois = rois[:T]

    x = torch.from_numpy(feats).unsqueeze(0)  # (1,T,D)
    r = torch.from_numpy(rois.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(2)  # (1,T,1,H,W)
    lengths = torch.tensor([T], dtype=torch.long)

    with torch.no_grad():
        log_probs_tbc = model(x, r, lengths)  # (T,1,C)

    lp_tc = log_probs_tbc[:T, 0, :]          # (T,C)

    scored = []
    best_lab, best_score = None, -1e18

    for lab, ids in dict_words:
        score = ctc_word_logprob(lp_tc, ids, blank=blank_id).item()

        # NEW: length penalty
        if exp_len and lab in exp_len and len_lambda > 0:
            mu = float(exp_len[lab])
            score = score - len_lambda * abs(T - mu)

        scored.append((lab, score))
        if score > best_score:
            best_score = score
            best_lab = lab

    scored.sort(key=lambda z: z[1], reverse=True)
    return best_lab, best_score, scored[:3]

# ---------------- camera + draw ----------------
def open_camera():
    for idx in [CAM_INDEX, 1, 0]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()
    raise RuntimeError("No camera found.")

def draw_lips_only(frame_bgr, result):
    out = frame_bgr.copy()

    if not result.face_landmarks:
        return out
    face = result.face_landmarks[0]
    h, w = out.shape[:2]

    def px(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    if DRAW_POINTS:
        for i in MOUTH_SET:
            cv2.circle(out, px(i), 1, (0, 255, 0), -1)
    return out

def main():
    if not os.path.exists(CTC_MODEL_PATH):
        raise RuntimeError(f"Missing {CTC_MODEL_PATH}. Train first.")

    model, dict_words, max_t, blank_id, vocab, exp_len, len_lambda = load_ctc_model_roi(CTC_MODEL_PATH)
    print("Loaded ROI+CTC model. max_t =", max_t, "len_lambda =", len_lambda)
    print("exp_len:", exp_len)

    cap = open_camera()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow("Lips Only", cv2.WINDOW_NORMAL)

    recording = False
    feats_buf = []
    roi_buf = []
    prev_xy_norm = None
    clip_id = 0

    t0 = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts_ms = int((time.monotonic() - t0) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            out = draw_lips_only(frame_bgr, result)

            if recording and result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = frame_bgr.shape[:2]
                feat, prev_xy_norm, center, mouth_w = extract_mouth_features(face, w, h, prev_xy_norm)
                roi = crop_mouth_roi_gray(frame_bgr, center, mouth_w)
                if roi is not None:
                    feats_buf.append(feat)
                    roi_buf.append(roi)
            elif not recording:
                prev_xy_norm = None

            status = "REC" if recording else "IDLE"
            cv2.putText(out, f"{status} | r start/stop | q quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if recording:
                cv2.putText(out, f"frames: {len(feats_buf)}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if last_pred:
                y0 = 110
                best_lab, top3 = last_pred
                cv2.putText(out, f"Best: {best_lab}",
                            (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                for k, (lab, score) in enumerate(top3):
                    cv2.putText(out, f"{k+1}) {lab}  {score:.1f}",
                                (20, y0 + 30*(k+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Lips Only", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            ch = chr(key) if 32 <= key < 127 else ""
            if ch in key_to_word:
                current_label = key_to_word[ch]

            if ch == "r":
                recording = not recording
                if recording:
                    feats_buf = []
                    roi_buf = []
                    prev_xy_norm = None
                else:
                    print(f"Recording stopped. frames={len(feats_buf)}. Predicting...")
                    if len(feats_buf) < 5:
                        print("Clip too short / no ROI.")
                        continue

                    feats = np.stack(feats_buf, axis=0).astype(np.float32)
                    rois  = np.stack(roi_buf, axis=0).astype(np.uint8)

                    feats, rois = trim_silence_pair_np(feats, rois, open_idx=-3, thresh=0.05, pad=2)

                    print("raw T", len(feats_buf), "trimmed T", feats.shape[0])


                    pred = predict_word_ctc_roi(model, dict_words, blank_id, feats, rois, max_t,
                            exp_len=exp_len, len_lambda=len_lambda)
                    if pred is None:
                        print("No prediction.")
                    else:
                        best_lab, best_score, top3 = pred
                        last_pred = (best_lab, top3)
                        print("Prediction:", best_lab, "Top3:", top3)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
