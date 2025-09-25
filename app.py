# import os, cv2, math, threading, re
# import numpy as np
# from io import BytesIO
# from typing import Optional, Dict, Any, Literal
# from collections import deque

# from fastapi import FastAPI, UploadFile, File, Query, Response
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.encoders import jsonable_encoder
# from pydantic import BaseModel
# import mediapipe as mp
# from PIL import Image, ImageOps


# HF_STICKY_N   = int(os.getenv("HF_STICKY_N", 5))  # history length
# HF_STICKY_MIN = int(os.getenv("HF_STICKY_MIN", 3))# min samples to enforce
# _hf_hist = deque(maxlen=HF_STICKY_N)
# # =========================
# # -------- Config ---------
# # =========================
# APP_TITLE = "Hybrid Patient Positioning"
# MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
# ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
# ORIENT_HEAD_TOWARD = os.getenv("ORIENT_HEAD_TOWARD", "TOP").upper() # "TOP" or "BOTTOM"
# YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s-pose.pt")
# IMGSZ = int(os.getenv("IMGSZ", "1536")) # slightly larger for more stable keypoints


# DECUB_AXIS_DEG_STRICT = float(os.getenv("DECUB_AXIS_DEG_STRICT", "25.0"))
# DECUB_LR_THRESH_STRICT = float(os.getenv("DECUB_LR_THRESH_STRICT", "0.60"))
# DECUB_LR_THRESH_RELAX = float(os.getenv("DECUB_LR_THRESH_RELAX", "0.12"))


# DEFAULT_SUPINE_IF_UNCERTAIN = int(os.getenv("DEFAULT_SUPINE_IF_UNCERTAIN", "1")) == 1
# SUPINE_UNKNOWN_AXIS_MAX = float(os.getenv("SUPINE_UNKNOWN_AXIS_MAX", "28.0"))
# SUPINE_UNKNOWN_MIN_LOWERKP = float(os.getenv("SUPINE_UNKNOWN_MIN_LOWERKP", "0.30"))


# DRAW_SKELETON = int(os.getenv("DRAW_SKELETON", "1")) == 1
# PROCESS_EVERY_N = int(os.getenv("PROCESS_EVERY_N", "2"))


# # =========================
# # ------ 3rd party --------
# # =========================
# from ultralytics import YOLO
# from mediapipe.tasks import python as mp_python
# from mediapipe.tasks.python import vision as mp_vision


# # Load MediaPipe Tasks models
# SEGMENTER_PATH = os.getenv("SEGMENTER_MODEL", "models/selfie_segmenter.tflite")
# FACEDET_PATH = os.getenv("FACEDET_MODEL", "models/blaze_face_short_range.tflite")


# BaseOptions = mp_python.BaseOptions


# # Segmentation
# seg_options = mp_vision.ImageSegmenterOptions(
# base_options=BaseOptions(model_asset_path=SEGMENTER_PATH),
# output_category_mask=True
# )
# mp_seg = mp_vision.ImageSegmenter.create_from_options(seg_options)


# # Face Detection
# fd_options = mp_vision.FaceDetectorOptions(
# base_options=BaseOptions(model_asset_path=FACEDET_PATH),
# min_detection_confidence=0.3
# )
# mp_fd = mp_vision.FaceDetector.create_from_options(fd_options)


# yolo = YOLO(YOLO_MODEL)
# HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# # =========================
# # ------- FastAPI ---------
# # =========================
# app = FastAPI(title=APP_TITLE)
# app.add_middleware(
# CORSMiddleware,
# allow_origins=ALLOW_ORIGINS,
# allow_credentials=True,
# allow_methods=["*"],
# allow_headers=["*"],
# )


# class AnalyzeOut(BaseModel):
#     position: str
#     label: Literal[
#         "head_first_supine", "head_first_prone",
#         "feet_first_supine", "feet_first_prone",
#         "head_first_decubitus_right", "head_first_decubitus_left",
#         "feet_first_decubitus_right", "feet_first_decubitus_left",
#         "unknown"
#     ]
#     confidence: float
#     why: str
#     debug: Optional[Dict[str, Any]] = None


# # =========================
# # ------ Utilities --------
# # =========================
# COCO_LINES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
# def _clip01(x): return float(max(0.0, min(1.0, x)))

# def _apply_headfeet_sticky(label: str):
#     """Debounce head/feet by majority vote over last N decisions."""
#     is_hf = label.startswith("head_first_")
#     _hf_hist.append(is_hf)
#     if len(_hf_hist) >= HF_STICKY_MIN:
#         maj = sum(_hf_hist) > (len(_hf_hist) / 2.0)
#         if is_hf != maj:
#             if maj:   # flip to Head First
#                 return label.replace("feet_first_", "head_first_"), "sticky_majority_flip"
#             else:     # flip to Feet First
#                 return label.replace("head_first_", "feet_first_"), "sticky_majority_flip"
#     return label, ""


# def preprocess(img):
#     h, w = img.shape[:2]
#     if max(h, w) > 1600:
#         s = 1600 / max(h, w)
#         img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
#     img = cv2.bilateralFilter(img, 5, 50, 50)
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:, :, 0])
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# def segment_person(img):
#     # img: BGR np.array
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # MediaPipe Tasks expects an mp.Image (from top-level mediapipe)
#     mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     # segment() (NOT process)
#     res = mp_seg.segment(mp_img)

#     # You asked for output_category_mask=True -> this is a label mask.
#     # For the Selfie Segmentation model, foreground/person is class 1.
#     if res is None or res.category_mask is None:
#         return None, 0.0

#     cat = res.category_mask.numpy_view()               # uint8 [H,W] of class ids
#     fg = (cat == 1).astype(np.uint8)                  # 1 where person
#     conf = float(np.mean(fg))                         # simple “confidence” = person area ratio

#     bin_mask = (fg * 255).astype(np.uint8)

#     # Keep only largest connected component
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
#     if num <= 1:
#         return None, conf
#     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     largest_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

#     return largest_mask, conf


# def pca_axis(mask):
#     ys, xs = np.nonzero(mask)
#     if len(xs) < 200:
#         return None, None, 0.0
#     pts = np.stack([xs, ys], axis=1).astype(np.float32)
#     mean = pts.mean(axis=0)
#     X = pts - mean
#     _, _, Vt = np.linalg.svd(X, full_matrices=False)
#     v = Vt[0]
#     proj = X @ v
#     p_min = pts[int(np.argmin(proj))]
#     p_max = pts[int(np.argmax(proj))]
#     # 90° means vertical; closer to 0° means horizontal
#     ang = abs(90.0 - abs(math.degrees(math.atan2(v[1], v[0]))))
#     return (mean, v), (p_min, p_max), float(ang)

# def detect_face_mediapipe(img, mask=None):
#     # img: BGR np.array
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     # detect() (NOT process)
#     res = mp_fd.detect(mp_img)
#     if res is None or not res.detections:
#         return None

#     H, W = img.shape[:2]
#     best = None
#     best_area = 0

#     for det in res.detections:
#         # Tasks API provides pixel-space bounding_box
#         bb = det.bounding_box
#         x = int(bb.origin_x)
#         y = int(bb.origin_y)
#         w = int(bb.width)
#         h = int(bb.height)

#         # clamp to image
#         x = max(0, min(W - 1, x))
#         y = max(0, min(H - 1, y))
#         w = max(1, min(W - x, w))
#         h = max(1, min(H - y, h))

#         # ensure it overlaps the person mask if provided
#         if mask is not None:
#             x1, y1, x2, y2 = x, y, min(W - 1, x + w), min(H - 1, y + h)
#             crop = mask[y1:y2, x1:x2]
#             if crop.size == 0 or np.count_nonzero(crop) == 0:
#                 continue

#         area = w * h
#         if area > best_area:
#             best_area = area
#             best = (x, y, w, h)

#     return best


# def detect_face(img, mask=None):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = HAAR.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(28,28), flags=cv2.CASCADE_SCALE_IMAGE)
#     best = None; area = 0
#     for (x,y,w,h) in faces:
#         if mask is not None:
#             x1,y1,x2,y2 = x,y,x+w,y+h
#             x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(mask.shape[1]-1,x2), min(mask.shape[0]-1,y2)
#             crop = mask[y1:y2, x1:x2]
#             if crop.size==0 or np.count_nonzero(crop)==0:
#                 continue
#         if w*h > area:
#             area = w*h; best = (x,y,w,h)
#     if best is not None:
#         return best
#     return detect_face_mediapipe(img, mask=mask)

# # =========================
# # --- Segmentation pass ---
# # =========================
# def classify_segmentation(img_bgr, want_debug=False):
#     img = preprocess(img_bgr)
#     mask, seg_conf = segment_person(img)
#     if mask is None:
#         out = {
#             "label": "unknown",
#             "position": "Unknown",
#             "confidence": 0.0,
#             "why": "segmentation_failed",
#         }
#         if want_debug:
#             out["debug"] = {"seg_conf": seg_conf}
#         return out

#     axis, ends, axis_vert = pca_axis(mask)
#     if axis is None:
#         return {"label":"unknown","position":"Unknown","confidence":0.2,"why":"small_mask"}

#     (pA, pB) = ends
#     face = detect_face(img, mask)
#     head_xy = None
#     face_vis = 0.0
#     if face is not None:
#         x,y,w,h = face
#         head_xy = np.array([x + w/2.0, y + h/2.0], dtype=np.float32)
#         face_vis = 1.0

#     if head_xy is not None:
#         dA = np.linalg.norm(head_xy - pA)
#         dB = np.linalg.norm(head_xy - pB)
#         head_end = pA if dA <= dB else pB
#         feet_end = pB if head_end is pA else pA
#     else:
#         head_end, feet_end = pA, pB

#     H, W = mask.shape
#     y_margin = max(8, int(0.03 * H))
#     if ORIENT_HEAD_TOWARD == "TOP":
#         head_first = (head_end[1] + y_margin < feet_end[1])
#     else:
#         head_first = (head_end[1] - y_margin > feet_end[1])

#     supine = (face_vis >= 0.5)

#     xs = np.nonzero(mask)[1]
#     left_mass  = np.sum(xs <  W * 0.48)
#     right_mass = np.sum(xs >= W * 0.52)
#     denom = max(1, left_mass + right_mass)
#     lr_imb = abs(left_mass - right_mass) / denom
#     lr_thresh = DECUB_LR_THRESH_STRICT if axis_vert < DECUB_AXIS_DEG_STRICT else DECUB_LR_THRESH_RELAX
#     is_side = lr_imb > lr_thresh
#     side_label = ("right" if left_mass > right_mass else "left") if is_side else None

#     if is_side:
#         if head_first:
#             label = f"head_first_decubitus_{side_label}"
#             text  = f"Head First Decubitus {'Right' if side_label=='right' else 'Left'}"
#         else:
#             label = f"feet_first_decubitus_{side_label}"
#             text  = f"Feet First Decubitus {'Right' if side_label=='right' else 'Left'}"
#     else:
#         if head_first and supine:        label, text = "head_first_supine",  "Head First Supine"
#         elif head_first and not supine:  label, text = "head_first_prone",   "Head First Prone"
#         elif not head_first and supine:  label, text = "feet_first_supine",  "Feet First Supine"
#         else:                            label, text = "feet_first_prone",   "Feet First Prone"

#     seg_q  = _clip01((np.count_nonzero(mask) / (H*W)) * 4.0)
#     axis_q = _clip01(1.0 - (axis_vert / 90.0))
#     face_q = face_vis
#     side_q = lr_imb if is_side else (1.0 - lr_imb)
#     conf = float(round(0.15 + 0.35*seg_q + 0.25*axis_q + 0.25*max(face_q, side_q), 3))

#     why = f"seg={seg_conf:.2f}, axis={axis_vert:.1f}, face={face_vis:.2f}, lr_imb={lr_imb:.2f}, head_first={head_first}"

#     # IMPORTANT: expose cues at TOP LEVEL (not only in debug)
#     out = {
#         "label": label,
#         "position": text,
#         "confidence": conf,
#         "why": why,
#         "mask": mask,
#         "face": face,
#         "head_end": head_end,
#         "feet_end": feet_end,
#         "img_pp": img,
#         "pca_ends": (pA, pB),

#         # top-level cues used by fusion/safeguards even when want_debug=False
#         "axis_vert": float(axis_vert),
#         "face_vis": float(face_vis),
#         "lr_thresh": float(lr_thresh),
#     }
#     if want_debug:
#         out["debug"] = {
#             "seg_conf": round(seg_conf, 3),
#             "axis_vert": round(axis_vert, 1),
#             "face_vis": face_vis,
#             "lr_thresh": lr_thresh,
#         }
#     return out

# # =========================
# # ----- YOLO fallback -----
# # (shoulders as supine backup, robust head/feet)
# # =========================
# def yolo_pose_classify(img_bgr):
#     """
#     Classify posture using YOLOv8-Pose keypoints only.
#     - Head/feet: prefer nose vs ankles; if nose missing, infer from ankles vs image midline.
#     - Supine/prone: nose present OR either shoulder confident -> supine.
#     """
#     res = yolo.predict(source=img_bgr, imgsz=IMGSZ, conf=0.2, iou=0.45, verbose=False)
#     r0 = res[0]
#     if len(r0.boxes) == 0 or r0.keypoints is None or r0.keypoints.xy is None:
#         return {"label": "unknown", "position": "Unknown", "confidence": 0.2, "why": "no_pose", "res": res}

#     kxy = r0.keypoints.xy[0].cpu().numpy()
#     kcf = r0.keypoints.conf[0].cpu().numpy()
#     H = img_bgr.shape[0]

#     def get(i):
#         if i < len(kxy):
#             x = float(kxy[i, 0]); y = float(kxy[i, 1]); c = float(kcf[i])
#             return x, y, (0.0 if math.isnan(c) else max(0.0, min(1.0, c)))
#         return float("nan"), float("nan"), 0.0

#     NOSE, LSHO, RSHO, LANK, RANK = 0, 5, 6, 15, 16
#     _, ny, nc   = get(NOSE)
#     _, _, scL = get(LSHO)
#     _, _, scR = get(RSHO)
#     _, ly, lc   = get(LANK)
#     _, ry, rc   = get(RANK)

#     feet_y = np.nanmean([ly, ry])

#     # ---- Head vs feet (SIGN NOW MATCHES SEGMENTATION BRANCH) ----
#     ORIENT = ORIENT_HEAD_TOWARD
#     head_first = None
#     if not (math.isnan(ny) or math.isnan(feet_y)):
#         if ORIENT == "TOP":
#             head_first = (ny < feet_y)       # head lower than feet
#         else:  # BOTTOM
#             head_first = (ny > feet_y)       # head higher than feet
#     elif not math.isnan(feet_y):
#         # Nose missing → fallback: compare ankles to image midline
#         if ORIENT == "TOP":
#             head_first = (feet_y >= (H * 0.5))   # feet lower half ⇒ head is up top ⇒ Head-First
#         else:  # BOTTOM
#             head_first = (feet_y <= (H * 0.5))   # feet upper half ⇒ head is down ⇒ Head-First
#     if head_first is None:
#         head_first = True

#     # ---- Supine vs prone ----
#     shoulders_good = (scL > 0.25) or (scR > 0.25)
#     supine = (nc >= 0.20) or shoulders_good

#     # ---- Label ----
#     if head_first and supine:
#         label, text = "head_first_supine", "Head First Supine"
#     elif head_first and not supine:
#         label, text = "head_first_prone", "Head First Prone"
#     elif not head_first and supine:
#         label, text = "feet_first_supine", "Feet First Supine"
#     else:
#         label, text = "feet_first_prone", "Feet First Prone"

#     # ---- Confidence & why ----
#     conf_terms = [v for v in [nc, scL, scR, lc, rc] if v > 0]
#     mean_c = float(np.mean(conf_terms)) if conf_terms else 0.0
#     conf = float(round(0.42 + 0.25 * min(1.0, mean_c), 3))
#     rule = "nose_vs_ankles" if not math.isnan(ny) else "ankles_center_fallback"
#     why = f"yolo_fallback({rule}) nose={nc:.2f} shoulders={(scL+scR)/2:.2f} ankles={(lc+rc)/2:.2f}"

#     return {"label": label, "position": text, "confidence": conf, "why": why, "res": res}

# def _override_head_first_with_ankles(seg, yolo_res):
#     """
#     Use YOLO ankle Y-positions to decide head/feet robustly.
#     Returns (head_first_bool_or_None, used_override_bool).
#     """
#     try:
#         rlist = yolo_res.get("res")
#         if not rlist:
#             return (None, False)
#         r0 = rlist[0]
#         if r0.keypoints is None or r0.keypoints.xy is None or len(r0.keypoints.xy) == 0:
#             return (None, False)

#         kxy = r0.keypoints.xy[0].cpu().numpy()
#         kcf = r0.keypoints.conf[0].cpu().numpy() if r0.keypoints.conf is not None else None

#         def xy(i):
#             try:
#                 x, y = float(kxy[i, 0]), float(kxy[i, 1])
#                 if math.isnan(x) or math.isnan(y):
#                     return None, None
#                 return x, y
#             except Exception:
#                 return None, None

#         def conf(i):
#             try:
#                 if kcf is None:
#                     return 0.0
#                 v = float(kcf[i])
#                 return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
#             except Exception:
#                 return 0.0

#         LANK, RANK = 15, 16
#         _, ly = xy(LANK)
#         _, ry = xy(RANK)
#         lc, rc = conf(LANK), conf(RANK)

#         # need ankle evidence
#         if ly is None and ry is None:
#             return (None, False)

#         mean_c = (lc + rc) / 2.0
#         if mean_c < 0.50:  # stricter than before (was 0.35) -> fewer spurious overrides
#             return (None, False)

#         feet_y = np.nanmean([v for v in [ly, ry] if v is not None])
#         ORIENT = ORIENT_HEAD_TOWARD

#         # Prefer segmentation endpoints if available (and apply the SAME sign as seg branch)
#         if seg.get("head_end") is not None and seg.get("feet_end") is not None:
#             he_y = float(seg["head_end"][1])
#             fe_y = float(seg["feet_end"][1])
#             head_first = (he_y > fe_y) if ORIENT == "TOP" else (he_y < fe_y)
#         else:
#             # derive image height
#             H = None
#             m = seg.get("mask")
#             if isinstance(m, np.ndarray):
#                 H = m.shape[0]
#             if H is None:
#                 try:
#                     H = int(r0.orig_shape[0]) if hasattr(r0, "orig_shape") else None
#                 except Exception:
#                     H = None
#             if H is None:
#                 return (None, False)

#             # if feet near midline, don't force an override
#             if abs(feet_y - (H * 0.5)) < (0.05 * H):  # within 5% of height from midline
#                 return (None, False)

#             head_first = (feet_y >= (H * 0.5)) if ORIENT == "TOP" else (feet_y <= (H * 0.5))

#         print(f"[CONFIG] ORIENT_HEAD_TOWARD = {ORIENT_HEAD_TOWARD}", flush=True)
#         return (head_first, True)
#     except Exception:
#         return (None, False)

# # =========================
# # ------- Visualization ---
# # =========================
# def draw_overlay(img_bgr, seg_result=None, yolo_result=None):
#     img = img_bgr.copy()
#     if seg_result and isinstance(seg_result.get("mask"), np.ndarray):
#         mask = seg_result["mask"]
#         cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(img, cnts, -1, (0,220,80), 2)
#         he, fe = seg_result.get("head_end"), seg_result.get("feet_end")
#         if he is not None and fe is not None:
#             cv2.circle(img, (int(he[0]), int(he[1])), 6, (0,255,255), -1)
#             cv2.circle(img, (int(fe[0]), int(fe[1])), 6, (255,200,0), -1)
#             cv2.line(img, (int(he[0]), int(he[1])), (int(fe[0]), int(fe[1])), (255,255,0), 2)
#         face = seg_result.get("face")
#         if face is not None:
#             x,y,w,h = face
#             cv2.rectangle(img, (x,y), (x+w,y+h), (0,128,255), 2)

#     if DRAW_SKELETON and yolo_result and yolo_result.get("res"):
#         r0 = yolo_result["res"][0]
#         if r0.keypoints is not None and r0.keypoints.xy is not None and len(r0.keypoints.xy) > 0:
#             kxy = r0.keypoints.xy[0].cpu().numpy()
#             for a,b in COCO_LINES:
#                 if a < len(kxy) and b < len(kxy):
#                     xa,ya = kxy[a]; xb,yb = kxy[b]
#                     if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
#                         cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0,255,0), 2)
#             for (x,y) in kxy:
#                 if not (np.isnan(x) or np.isnan(y)):
#                     cv2.circle(img, (int(x),int(y)), 3, (0,200,255), -1)
#     return img

# # =========================
# # ------- Helpers ---------
# # =========================
# def _ankle_centroid_y(yolo_res):
#     """Return (centroid_y, conf_mean) for ankles from YOLO result, or (nan, 0.0)."""
#     try:
#         r0 = yolo_res.get("res")[0]
#         kxy = r0.keypoints.xy[0].cpu().numpy()
#         kcf = r0.keypoints.conf[0].cpu().numpy()
#         LANK, RANK = 15, 16
#         ys, cs = [], []
#         for i in (LANK, RANK):
#             if i < len(kxy):
#                 y = float(kxy[i,1]); c = float(max(kcf[i], 0.0))
#                 if not (math.isnan(y) or math.isnan(c)) and c > 0.0:
#                     ys.append(y); cs.append(c)
#         if not cs:
#             return float("nan"), 0.0
#         return float(np.average(ys, weights=cs)), float(np.mean(cs))
#     except Exception:
#         return float("nan"), 0.0

# # def _override_head_first_with_ankles(seg, yolo_res):
# #     """
# #     Use YOLO ankle Y-positions to decide head/feet robustly.
# #     Returns (head_first_bool_or_None, used_override_bool).

# #     Works even if segmentation failed (no mask), by reading image height
# #     from YOLO's result or falling back to seg endpoints when present.
# #     """
# #     try:
# #         rlist = yolo_res.get("res")
# #         if not rlist:
# #             return (None, False)
# #         r0 = rlist[0]
# #         if r0.keypoints is None or r0.keypoints.xy is None or len(r0.keypoints.xy) == 0:
# #             return (None, False)

# #         kxy = r0.keypoints.xy[0].cpu().numpy()
# #         kcf = r0.keypoints.conf[0].cpu().numpy() if r0.keypoints.conf is not None else None

# #         def xy(i):
# #             try:
# #                 x, y = float(kxy[i, 0]), float(kxy[i, 1])
# #                 if math.isnan(x) or math.isnan(y):
# #                     return None, None
# #                 return x, y
# #             except Exception:
# #                 return None, None

# #         def conf(i):
# #             try:
# #                 if kcf is None: return 0.0
# #                 v = float(kcf[i]); 
# #                 return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
# #             except Exception:
# #                 return 0.0

# #         LANK, RANK = 15, 16
# #         _, ly = xy(LANK)
# #         _, ry = xy(RANK)
# #         lc, rc = conf(LANK), conf(RANK)

# #         # need ankle evidence
# #         if ly is None and ry is None:
# #             return (None, False)
# #         mean_c = (lc + rc) / 2.0
# #         if mean_c < 0.35:
# #             return (None, False)

# #         # prefer seg endpoints if available
# #         head_first = None
# #         ORIENT = os.getenv("ORIENT_HEAD_TOWARD", "BOTTOM").upper()
# #         feet_y = np.nanmean([v for v in [ly, ry] if v is not None])

# #         if seg.get("head_end") is not None and seg.get("feet_end") is not None:
# #             he_y = float(seg["head_end"][1]); fe_y = float(seg["feet_end"][1])
# #             head_first = (he_y > fe_y) if ORIENT == "BOTTOM" else (he_y < fe_y)
# #         else:
# #             # compare ankle centroid to frame center height
# #             H = None
# #             m = seg.get("mask")
# #             if isinstance(m, np.ndarray):
# #                 H = m.shape[0]
# #             if H is None:
# #                 try:
# #                     H = int(r0.orig_shape[0]) if hasattr(r0, "orig_shape") else None
# #                 except Exception:
# #                     H = None

# #             if H is not None:
# #                if ORIENT_HEAD_TOWARD == "TOP":
# #     # Feet above midline (smaller y) => Feet First (head_first=False)
# #                     head_first = not (feet_y < (H * 0.5))
# #             else:  # ORIENT_HEAD_TOWARD == "BOTTOM"
# #     # Feet below midline (larger y) => Feet First (head_first=False)
# #                     head_first = not (feet_y > (H * 0.5))




# #         return (head_first, True) if head_first is not None else (None, False)
# #     except Exception:
# #         return (None, False)

# def _is_decubitus(lbl: str) -> bool:
#     return "decubitus" in (lbl or "")

# # =========================
# # ------- Fusion  ---------
# # =========================

# def _ankle_confidence(yolo_res):
#     """
#     Return (left_conf, right_conf, mean_conf) for ankles from YOLO result,
#     or (0.0, 0.0, 0.0) if unavailable.
#     """
#     try:
#         r0 = yolo_res.get("res")[0]
#         kcf = r0.keypoints.conf[0].cpu().numpy()
#         LANK, RANK = 15, 16
#         lc = float(kcf[LANK]) if LANK < len(kcf) and not math.isnan(float(kcf[LANK])) else 0.0
#         rc = float(kcf[RANK]) if RANK < len(kcf) and not math.isnan(float(kcf[RANK])) else 0.0
#         meanc = float(np.nanmean([lc, rc])) if (lc > 0.0 or rc > 0.0) else 0.0
#         return lc, rc, meanc
#     except Exception:
#         return 0.0, 0.0, 0.0


# def _nose_shoulder_confidence(yolo_res):
#     """
#     Returns (nose_conf, shoulders_mean_conf, ankles_mean_conf) in [0..1].
#     Always returns a 3-tuple, never None.
#     """
#     try:
#         rlist = yolo_res.get("res")
#         if not rlist or len(rlist) == 0:
#             return 0.0, 0.0, 0.0
#         r0 = rlist[0]
#         if r0.keypoints is None or r0.keypoints.conf is None or len(r0.keypoints.conf) == 0:
#             return 0.0, 0.0, 0.0

#         kcf = r0.keypoints.conf[0].cpu().numpy()
#         def g(i):
#             try:
#                 v = float(kcf[i])
#                 return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
#             except Exception:
#                 return 0.0

#         nose      = g(0)
#         shoulders = (g(5) + g(6)) / 2.0   # L-shoulder, R-shoulder
#         ankles    = (g(15) + g(16)) / 2.0 # L-ankle, R-ankle
#         return nose, shoulders, ankles
#     except Exception:
#         return 0.0, 0.0, 0.0


# def decide_hybrid(img, want_debug=False):
#     """
#     Fuse Segmentation (PCA + face cue) with YOLOv8-Pose.
#     Applies ankle override, supine safeguards, and head/feet debouncing.
#     """
#     # ---- run both branches ----
#     seg = classify_segmentation(img, want_debug=want_debug)
#     yolo_res = yolo_pose_classify(img)

#     # ---- initial winner ----
#     best = seg if seg.get("confidence", 0.0) >= yolo_res.get("confidence", 0.0) else yolo_res

#     # ---- diagnostics from seg ----
#     seg_dbg   = seg.get("debug") or {}
#     axis_vert = seg.get("axis_vert", seg_dbg.get("axis_vert"))
#     face_vis  = seg.get("face_vis",  seg_dbg.get("face_vis", 0.0))
#     is_decub  = ("decubitus" in (seg.get("label") or ""))

#     # (1) Soften decubitus if axis is very straight (likely not true side)
#     if is_decub and axis_vert is not None and axis_vert < DECUB_AXIS_DEG_STRICT:
#         seg = dict(seg)
#         seg["confidence"] *= 0.6
#         if best is seg:
#             best = seg

#     # (2) Override head/feet using ankle evidence (robust when face/mask fail)
#     hf_new, used_override = _override_head_first_with_ankles(seg, yolo_res)
#     print(f"[FUSION] override={used_override}, hf_new={hf_new}", flush=True)

#     def _flip(label: str, pos: str, head_first_now: bool):
#         if head_first_now and label.startswith("feet_first_"):
#             return label.replace("feet_first_", "head_first_"), pos.replace("Feet First", "Head First")
#         if (not head_first_now) and label.startswith("head_first_"):
#             return label.replace("head_first_", "feet_first_"), pos.replace("Head First", "Feet First")
#         return label, pos

#     if used_override and hf_new is not None:
#         # apply to 'best'
#         b_lab, b_pos = _flip(best.get("label", ""), best.get("position", ""), hf_new)
#         if (b_lab, b_pos) != (best.get("label"), best.get("position")):
#             best = dict(best)
#             best["label"], best["position"] = b_lab, b_pos
#             best["why"] = (best.get("why") or "") + " | applied_ankle_override"

#         # keep seg consistent for overlay
#         s_lab, s_pos = _flip(seg.get("label", ""), seg.get("position", ""), hf_new)
#         if (s_lab, s_pos) != (seg.get("label"), seg.get("position")):
#             seg = dict(seg)
#             seg["label"], seg["position"] = s_lab, s_pos

#     # (3) Prefer YOLO if seg says decubitus but YOLO confidently says plain posture
#     if is_decub and "decubitus" not in (yolo_res.get("label") or ""):
#         if yolo_res.get("confidence", 0.0) >= 0.55 and seg.get("confidence", 0.0) <= 0.70:
#             best = yolo_res

#     # (4) If both are plain and agree, blend confidence for stability
#     if "decubitus" not in (seg.get("label") or "") and "decubitus" not in (yolo_res.get("label") or ""):
#         if seg.get("label") == yolo_res.get("label"):
#             best = dict(best)
#             best["confidence"] = float(min(1.0, 0.5 * seg["confidence"] + 0.5 * yolo_res["confidence"]))

#     # (5) Strong supine safeguards (work even if segmentation failed)
#     def _nose_shoulders_ankles(yolo_res_dict):
#         try:
#             r0 = yolo_res_dict.get("res")[0]
#             kcf = r0.keypoints.conf[0].cpu().numpy()
#             def g(i):
#                 try:
#                     v = float(kcf[i])
#                     return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
#                 except Exception:
#                     return 0.0
#             nose = g(0)
#             shoulders = (g(5) + g(6)) / 2.0
#             ankles = (g(15) + g(16)) / 2.0
#             return nose, shoulders, ankles
#         except Exception:
#             return 0.0, 0.0, 0.0

#     nose_c, shoulders_c, mean_ank = _nose_shoulders_ankles(yolo_res)

#     if best.get("label") in ("head_first_prone", "feet_first_prone"):
#         flipped = False

#         # A) Face OR any facial key (nose) seen -> force supine (highest priority)
#         if face_vis >= 0.5 or nose_c >= 0.20:
#             best = dict(best)
#             best["label"]      = best["label"].replace("prone", "supine")
#             best["position"]   = best["position"].replace("Prone", "Supine")
#             best["confidence"] = float(max(best.get("confidence", 0.55), 0.80))
#             best["why"]        = (best.get("why") or "") + " | supine_override(face/nose_detected)"
#             flipped = True

#         # B) If no face: allow override when axis straight OR seg failed, with ankle/shoulder support
#         axis_ok = (axis_vert is None) or (axis_vert < 25.0)
#         if not flipped and axis_ok and (mean_ank >= 0.40 or shoulders_c >= 0.25):
#             best = dict(best)
#             best["label"]      = best["label"].replace("prone", "supine")
#             best["position"]   = best["position"].replace("Prone", "Supine")
#             best["confidence"] = float(max(best.get("confidence", 0.55), 0.75))
#             best["why"]        = (best.get("why") or "") + " | supine_override(straight_axis_or_no_seg+ankles/shoulders)"

#     # (6) Final head/feet debouncing (sticky majority)
#     new_lab, tag = _apply_headfeet_sticky(best["label"])
#     if tag:
#         best = dict(best)
#         if new_lab.startswith("head_first_"):
#             best["position"] = best["position"].replace("Feet First", "Head First")
#         else:
#             best["position"] = best["position"].replace("Head First", "Feet First")
#         best["label"] = new_lab
#         best["why"] = (best.get("why") or "") + " | " + tag

#     return best, seg, yolo_res

# # =========================
# # ------- API -------------
# # =========================
# def _draw_title(vis, text: str):
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#     pad = 12
#     cv2.rectangle(vis, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), (0, 0, 0), -1)
#     cv2.putText(vis, text, (10 + pad, 10 + th + pad),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# def decode_image_exif_aware(data: bytes, content_type: Optional[str]) -> np.ndarray | None:
#     """Decode bytes -> BGR np.array, applying EXIF orientation for JPEGs."""
#     try:
#         ct = (content_type or "").lower()
#         if ct in ("image/jpeg", "image/jpg"):
#             pil = Image.open(BytesIO(data))
#             pil = ImageOps.exif_transpose(pil)  # apply EXIF rotation
#             rgb = np.array(pil)                 # RGB
#             if rgb.ndim == 3 and rgb.shape[2] == 3:
#                 return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#             if rgb.ndim == 2:
#                 return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
#     except Exception:
#         pass
#     # fallback to OpenCV
#     arr = np.frombuffer(data, np.uint8)
#     return cv2.imdecode(arr, cv2.IMREAD_COLOR)





# @app.get("/health")
# def health():
#     return {"ok": True, "service": APP_TITLE, "orient_head_toward": ORIENT_HEAD_TOWARD, "yolo_model": YOLO_MODEL}

# @app.post("/analyze", response_model=AnalyzeOut)
# async def analyze(image: UploadFile = File(...), debug: int = Query(0)):
#     try:
#         # basic validation
#         if not (image.content_type or "").startswith("image/"):
#             return JSONResponse({"error": "unsupported_media_type"}, status_code=415)

#         data = await image.read()
#         if len(data) > MAX_IMAGE_BYTES:
#             return JSONResponse({"error": "payload_too_large"}, status_code=413)

#         # EXIF-aware decode (handles phone JPEG rotations; falls back to OpenCV for PNG, etc.)
#         img = decode_image_exif_aware(data, image.content_type)
#         if img is None:
#             return JSONResponse({"error": "invalid_image"}, status_code=400)

#         # run hybrid decision
#         best, seg, yolo_res = decide_hybrid(img, want_debug=bool(debug))

#         dbg = None
#         if bool(debug):
#             dbg = {
#                 "seg_conf": seg.get("debug", {}).get("seg_conf") if seg.get("debug") else None,
#                 "axis_vert": seg.get("debug", {}).get("axis_vert") if seg.get("debug") else None,
#                 "face_vis":  seg.get("debug", {}).get("face_vis")  if seg.get("debug") else None,
#                 "lr_thresh": seg.get("debug", {}).get("lr_thresh") if seg.get("debug") else None,
#                 "why_seg": seg.get("why"),
#                 "why_yolo": yolo_res.get("why") if yolo_res else None,
#             }

#         out = {
#             "position":   best["position"],
#             "label":      best["label"],
#             "confidence": float(best["confidence"]),
#             "why":        best["why"],
#             **({"debug": dbg} if dbg else {}),
#         }
#         return JSONResponse(content=jsonable_encoder(out))
#     except Exception as e:
#         return JSONResponse({"error": f"server_error: {type(e).__name__}: {str(e)}"}, status_code=500)


# @app.post("/overlay")
# async def overlay(image: UploadFile = File(...)):
#     try:
#         data = await image.read()

#         # EXIF-aware decode
#         img = decode_image_exif_aware(data, image.content_type)
#         if img is None:
#             return JSONResponse({"error": "invalid_image"}, status_code=400)

#         # run + draw
#         best, seg, yolo_res = decide_hybrid(img, want_debug=False)
#         vis = draw_overlay(seg.get("img_pp", img), seg_result=seg, yolo_result=yolo_res)

#         label_txt = f"{best['position']} ({int(best['confidence']*100)}%)"
#         (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#         pad = 12
#         cv2.rectangle(vis, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), (0, 0, 0), -1)
#         cv2.putText(vis, label_txt, (10 + pad, 10 + th + pad),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#         ok, png = cv2.imencode(".png", vis)
#         if not ok:
#             return JSONResponse({"error": "encode_failed"}, status_code=500)

#         resp = Response(content=png.tobytes(), media_type="image/png")
#         resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
#         resp.headers["Pragma"] = "no-cache"
#         return resp
#     except Exception as e:
#         return JSONResponse({"error": f"server_error: {type(e).__name__}: {str(e)}"}, status_code=500)




# # ================ Optional webcam demo ================
# _cam = None
# _cam_lock = threading.Lock()
# _live_label = {"position":"—","label":"unknown","confidence":0.0,"why":"init"}
# _frame_idx = 0

# def get_cam():
#     global _cam
#     with _cam_lock:
#         if _cam is None:
#             _cam = cv2.VideoCapture(0)
#             _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         return _cam

# def gen_live_mjpeg():
#     global _frame_idx, _live_label
#     cap = get_cam()
#     while True:
#         with _cam_lock:
#             ok = cap.isOpened()
#         if not ok: break
#         ok, frame = cap.read()
#         if not ok: break

#         if _frame_idx % PROCESS_EVERY_N == 0:
#             best, seg, yolo_res = decide_hybrid(frame, want_debug=False)
#             _live_label = {
#                 "position": best["position"], "label": best["label"],
#                 "confidence": float(best["confidence"]), "why": best["why"]
#             }
#             vis_src = seg.get("img_pp", frame)
#             vis = draw_overlay(vis_src, seg_result=seg, yolo_result=yolo_res)
#         else:
#             vis = frame.copy()
#             txt = f"{_live_label['position']} ({int(_live_label['confidence']*100)}%)"
#             (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#             cv2.rectangle(vis, (10,10), (10+tw+24, 10+th+24), (0,0,0), -1)
#             cv2.putText(vis, txt, (22, 10+th+16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

#         _frame_idx += 1
#         ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
#         if not ok: continue
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

# @app.get("/live_mjpeg")
# def live_mjpeg():
#     get_cam()
#     return StreamingResponse(gen_live_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

# @app.get("/live_status")
# def live_status():
#     return JSONResponse(content=jsonable_encoder(_live_label))

# @app.on_event("shutdown")
# def _shutdown():
#     global _cam
#     with _cam_lock:
#         try:
#             if _cam is not None:
#                 _cam.release()
#         finally:
#             _cam = None
import os, cv2, math, threading, re
import numpy as np
from io import BytesIO
from typing import Optional, Dict, Any, Literal
from collections import deque

from fastapi import FastAPI, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import mediapipe as mp
from PIL import Image, ImageOps
import threading
from typing import Optional
import logging
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

HF_STICKY_N   = int(os.getenv("HF_STICKY_N", 5))   # history length
HF_STICKY_MIN = int(os.getenv("HF_STICKY_MIN", 3)) # min samples to enforce
_hf_hist = deque(maxlen=HF_STICKY_N)
ORIENT_LOCK = threading.Lock()


# =========================
# -------- Config ---------
# =========================
APP_TITLE = "Hybrid Patient Positioning"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
ORIENT_HEAD_TOWARD = os.getenv("ORIENT_HEAD_TOWARD", "TOP").upper()  # "TOP" or "BOTTOM"
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s-pose.pt")
IMGSZ = int(os.getenv("IMGSZ", "1536"))

logging.basicConfig(level=logging.INFO)

DECUB_AXIS_DEG_STRICT   = float(os.getenv("DECUB_AXIS_DEG_STRICT", "25.0"))
DECUB_LR_THRESH_STRICT  = float(os.getenv("DECUB_LR_THRESH_STRICT", "0.60"))
DECUB_LR_THRESH_RELAX   = float(os.getenv("DECUB_LR_THRESH_RELAX",  "0.12"))

DEFAULT_SUPINE_IF_UNCERTAIN = int(os.getenv("DEFAULT_SUPINE_IF_UNCERTAIN", "1")) == 1
SUPINE_UNKNOWN_AXIS_MAX     = float(os.getenv("SUPINE_UNKNOWN_AXIS_MAX", "28.0"))
SUPINE_UNKNOWN_MIN_LOWERKP  = float(os.getenv("SUPINE_UNKNOWN_MIN_LOWERKP","0.30"))

DRAW_SKELETON   = int(os.getenv("DRAW_SKELETON", "1")) == 1
PROCESS_EVERY_N = int(os.getenv("PROCESS_EVERY_N", "2"))

# =========================
# ------ 3rd party --------
# =========================
from ultralytics import YOLO
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Load MediaPipe Tasks models
SEGMENTER_PATH = os.getenv("SEGMENTER_MODEL", "models/selfie_segmenter.tflite")
FACEDET_PATH   = os.getenv("FACEDET_MODEL",   "models/blaze_face_short_range.tflite")

BaseOptions = mp_python.BaseOptions

# Segmentation
seg_options = mp_vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=SEGMENTER_PATH),
    output_category_mask=True
)
mp_seg = mp_vision.ImageSegmenter.create_from_options(seg_options)

# Face Detection
fd_options = mp_vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACEDET_PATH),
    min_detection_confidence=0.3
)
mp_fd = mp_vision.FaceDetector.create_from_options(fd_options)

yolo = YOLO(YOLO_MODEL)
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =========================
# ------- FastAPI ---------
# =========================
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeOut(BaseModel):
    position: str
    label: Literal[
        "head_first_supine","head_first_prone",
        "feet_first_supine","feet_first_prone",
        "head_first_decubitus_right","head_first_decubitus_left",
        "feet_first_decubitus_right","feet_first_decubitus_left",
        "unknown"
    ]
    confidence: float
    why: str
    debug: Optional[Dict[str, Any]] = None

# =========================
# ------ Utilities --------
# =========================
COCO_LINES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
def _clip01(x): return float(max(0.0, min(1.0, x)))

def _apply_headfeet_sticky(label: str):
    """Debounce head/feet by majority vote over last N decisions."""
    is_hf = label.startswith("head_first_")
    _hf_hist.append(is_hf)
    if len(_hf_hist) >= HF_STICKY_MIN:
        maj = sum(_hf_hist) > (len(_hf_hist) / 2.0)
        if is_hf != maj:
            if maj:
                return label.replace("feet_first_", "head_first_"), "sticky_majority_flip"
            else:
                return label.replace("head_first_", "feet_first_"), "sticky_majority_flip"
    return label, ""


def _norm_orient(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip().upper()
    if s in ("TOP", "BOTTOM"):
        return s
    # common misspellings / aliases
    if s in ("BUTTOM", "BOT", "DOWN"):
        return "BOTTOM"
    if s in ("UP",):
        return "TOP"
    return None

def preprocess(img):
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def segment_person(img):
    # img: BGR np.array
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # top-level mp.Image
    res = mp_seg.segment(mp_img)  # tasks API uses .segment()
    if res is None or res.category_mask is None:
        return None, 0.0
    cat = res.category_mask.numpy_view()        # uint8 [H,W] of class ids
    fg  = (cat == 1).astype(np.uint8)          # 1 where person
    conf = float(np.mean(fg))
    bin_mask = (fg * 255).astype(np.uint8)

    # Keep only largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return None, conf
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return largest_mask, conf

def pca_axis(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None, None, 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    X = pts - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    proj = X @ v
    p_min = pts[int(np.argmin(proj))]
    p_max = pts[int(np.argmax(proj))]
    # 90° means vertical; closer to 0° means horizontal
    ang = abs(90.0 - abs(math.degrees(math.atan2(v[1], v[0]))))
    return (mean, v), (p_min, p_max), float(ang)

def detect_face_mediapipe(img, mask=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = mp_fd.detect(mp_img)  # tasks API uses .detect()
    if res is None or not res.detections:
        return None
    H, W = img.shape[:2]
    best = None
    best_area = 0
    for det in res.detections:
        bb = det.bounding_box
        x = int(bb.origin_x); y = int(bb.origin_y)
        w = int(bb.width);    h = int(bb.height)
        x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        if mask is not None:
            x1, y1, x2, y2 = x, y, min(W - 1, x + w), min(H - 1, y + h)
            crop = mask[y1:y2, x1:x2]
            if crop.size == 0 or np.count_nonzero(crop) == 0:
                continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best

def detect_face(img, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = HAAR.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(28,28), flags=cv2.CASCADE_SCALE_IMAGE)
    best = None; area = 0
    for (x,y,w,h) in faces:
        if mask is not None:
            x1,y1,x2,y2 = x,y,x+w,y+h
            x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(mask.shape[1]-1,x2), min(mask.shape[0]-1,y2)
            crop = mask[y1:y2, x1:x2]
            if crop.size==0 or np.count_nonzero(crop)==0:
                continue
        if w*h > area:
            area = w*h; best = (x,y,w,h)
    if best is not None:
        return best
    return detect_face_mediapipe(img, mask=mask)

# =========================
# --- Segmentation pass ---
# =========================
def classify_segmentation(img_bgr, want_debug=False):
    img = preprocess(img_bgr)
    mask, seg_conf = segment_person(img)
    if mask is None:
        out = {"label":"unknown","position":"Unknown","confidence":0.0,"why":"segmentation_failed"}
        if want_debug: out["debug"] = {"seg_conf": seg_conf}
        return out

    axis, ends, axis_vert = pca_axis(mask)
    if axis is None:
        return {"label":"unknown","position":"Unknown","confidence":0.2,"why":"small_mask"}

    (pA, pB) = ends
    face = detect_face(img, mask)
    head_xy = None
    face_vis = 0.0
    if face is not None:
        x,y,w,h = face
        head_xy = np.array([x + w/2.0, y + h/2.0], dtype=np.float32)
        face_vis = 1.0

    if head_xy is not None:
        dA = np.linalg.norm(head_xy - pA)
        dB = np.linalg.norm(head_xy - pB)
        head_end = pA if dA <= dB else pB
        feet_end = pB if (head_end is pA) else pA
    else:
        head_end, feet_end = pA, pB

    H, W = mask.shape
    y_margin = max(8, int(0.03 * H))
    if ORIENT_HEAD_TOWARD == "TOP":
        head_first = (head_end[1] + y_margin < feet_end[1])
    else:
        head_first = (head_end[1] - y_margin > feet_end[1])

    supine = (face_vis >= 0.5)

    xs = np.nonzero(mask)[1]
    left_mass  = np.sum(xs <  W * 0.48)
    right_mass = np.sum(xs >= W * 0.52)
    denom = max(1, left_mass + right_mass)
    lr_imb = abs(left_mass - right_mass) / denom
    lr_thresh = DECUB_LR_THRESH_STRICT if axis_vert < DECUB_AXIS_DEG_STRICT else DECUB_LR_THRESH_RELAX
    is_side = lr_imb > lr_thresh
    side_label = ("right" if left_mass > right_mass else "left") if is_side else None

    if is_side:
        if head_first:
            label = f"head_first_decubitus_{side_label}"
            text  = f"Head First Decubitus {'Right' if side_label=='right' else 'Left'}"
        else:
            label = f"feet_first_decubitus_{side_label}"
            text  = f"Feet First Decubitus {'Right' if side_label=='right' else 'Left'}"
    else:
        if head_first and supine:        label, text = "head_first_supine",  "Head First Supine"
        elif head_first and not supine:  label, text = "head_first_prone",   "Head First Prone"
        elif not head_first and supine:  label, text = "feet_first_supine",  "Feet First Supine"
        else:                            label, text = "feet_first_prone",   "Feet First Prone"

    seg_q  = _clip01((np.count_nonzero(mask) / (H*W)) * 4.0)
    axis_q = _clip01(1.0 - (axis_vert / 90.0))
    face_q = face_vis
    side_q = lr_imb if is_side else (1.0 - lr_imb)
    conf = float(round(0.15 + 0.35*seg_q + 0.25*axis_q + 0.25*max(face_q, side_q), 3))

    why = f"seg={seg_conf:.2f}, axis={axis_vert:.1f}, face={face_vis:.2f}, lr_imb={lr_imb:.2f}, head_first={head_first}"

    out = {
        "label": label,
        "position": text,
        "confidence": conf,
        "why": why,
        "mask": mask,
        "face": face,
        "head_end": head_end,
        "feet_end": feet_end,
        "img_pp": img,
        "pca_ends": (pA, pB),
        "axis_vert": float(axis_vert),
        "face_vis": float(face_vis),
        "lr_thresh": float(lr_thresh),
    }
    if want_debug:
        out["debug"] = {
            "seg_conf": round(seg_conf, 3),
            "axis_vert": round(axis_vert, 1),
            "face_vis": face_vis,
            "lr_thresh": lr_thresh,
        }
    return out

# =========================
# ----- YOLO fallback -----
# # =========================
# def yolo_pose_classify(img_bgr):
#     res = yolo.predict(source=img_bgr, imgsz=IMGSZ, conf=0.2, iou=0.45, verbose=False)
#     r0 = res[0]
#     if len(r0.boxes) == 0 or r0.keypoints is None or r0.keypoints.xy is None:
#         return {"label": "unknown", "position": "Unknown", "confidence": 0.2, "why": "no_pose", "res": res}

#     kxy = r0.keypoints.xy[0].cpu().numpy()
#     kcf = r0.keypoints.conf[0].cpu().numpy()
#     H = img_bgr.shape[0]

#     def get(i):
#         if i < len(kxy):
#             x = float(kxy[i, 0]); y = float(kxy[i, 1]); c = float(kcf[i])
#             return x, y, (0.0 if math.isnan(c) else max(0.0, min(1.0, c)))
#         return float("nan"), float("nan"), 0.0

#     NOSE, LSHO, RSHO, LANK, RANK = 0, 5, 6, 15, 16
#     _, ny, nc = get(NOSE)
#     _, _, scL = get(LSHO)
#     _, _, scR = get(RSHO)
#     _, ly, lc = get(LANK)
#     _, ry, rc = get(RANK)

#     feet_y = np.nanmean([ly, ry])

#     # SIGN matches segmentation:
#     if ORIENT_HEAD_TOWARD == "TOP":
#         head_first = (ny < feet_y) if not (math.isnan(ny) or math.isnan(feet_y)) else None
#     else:
#         head_first = (ny > feet_y) if not (math.isnan(ny) or math.isnan(feet_y)) else None

#     if head_first is None and not math.isnan(feet_y):
#         if ORIENT_HEAD_TOWARD == "TOP":
#             head_first = (feet_y >= (H * 0.5))
#         else:
#             head_first = (feet_y <= (H * 0.5))
#     if head_first is None:
#         head_first = True

#     shoulders_good = (scL > 0.25) or (scR > 0.25)
#     supine = (nc >= 0.20) or shoulders_good

#     if head_first and supine:
#         label, text = "head_first_supine", "Head First Supine"
#     elif head_first and not supine:
#         label, text = "head_first_prone", "Head First Prone"
#     elif not head_first and supine:
#         label, text = "feet_first_supine", "Feet First Supine"
#     else:
#         label, text = "feet_first_prone", "Feet First Prone"

#     conf_terms = [v for v in [nc, scL, scR, lc, rc] if v > 0]
#     mean_c = float(np.mean(conf_terms)) if conf_terms else 0.0
#     conf = float(round(0.42 + 0.25 * min(1.0, mean_c), 3))
#     rule = "nose_vs_ankles" if not math.isnan(ny) else "ankles_center_fallback"
#     why = f"yolo_fallback({rule}) nose={nc:.2f} shoulders={(scL+scR)/2:.2f} ankles={(lc+rc)/2:.2f}"
#     return {"label": label, "position": text, "confidence": conf, "why": why, "res": res}
def yolo_pose_classify(img_bgr):
    """
    Classify posture using YOLOv8-Pose keypoints.
    - Head vs feet: nose vs ankles; fallback to ankles vs image midline.
    - Supine vs prone: nose/eyes visible => supine; else prone.
    """

    res = yolo.predict(source=img_bgr, imgsz=IMGSZ, conf=0.2, iou=0.45, verbose=False)
    r0 = res[0]

    if len(r0.boxes) == 0 or r0.keypoints is None or r0.keypoints.xy is None:
        return {
            "label": "unknown",
            "position": "Unknown",
            "confidence": 0.2,
            "why": "no_pose",
            "res": res,
        }

    kxy = r0.keypoints.xy[0].cpu().numpy()
    kcf = r0.keypoints.conf[0].cpu().numpy()
    H = img_bgr.shape[0]

    def get(i):
        if i < len(kxy):
            x = float(kxy[i, 0])
            y = float(kxy[i, 1])
            c = float(kcf[i])
            return x, y, (0.0 if math.isnan(c) else max(0.0, min(1.0, c)))
        return float("nan"), float("nan"), 0.0

    NOSE, LEYE, REYE, LSHO, RSHO, LANK, RANK = 0, 1, 2, 5, 6, 15, 16

    _, ny, nc = get(NOSE)
    _, _, le = get(LEYE)
    _, _, re = get(REYE)
    _, _, scL = get(LSHO)
    _, _, scR = get(RSHO)
    _, ly, lc = get(LANK)
    _, ry, rc = get(RANK)

    feet_y = np.nanmean([ly, ry])

    # orientation setting (default to TOP if not provided)
    ORIENT = os.getenv("ORIENT_HEAD_TOWARD", "TOP").upper()

    # ---- Head vs feet ----
    head_first = None
    if not (math.isnan(ny) or math.isnan(feet_y)):
        if ORIENT == "TOP":
            head_first = ny > feet_y  # head lower in image
        else:  # BOTTOM
            head_first = ny < feet_y  # head higher in image
    elif not math.isnan(feet_y):
        # Nose missing → fallback: compare ankles to image midline
        if ORIENT == "TOP":
            head_first = feet_y >= (H * 0.5)
        else:  # BOTTOM
            head_first = feet_y <= (H * 0.5)
    else:
        head_first = True  # default

    # ---- Supine vs prone ----
    eyes_good = (le > 0.3) or (re > 0.3)
    shoulders_good = (scL > 0.25) or (scR > 0.25)

    # new rule: nose or eyes present → supine; else prone
    supine = (nc > 0.3) or eyes_good
    if not supine and shoulders_good and nc < 0.2:
        # shoulders visible but no face → more likely prone
        supine = False

    # ---- Label ----
    if head_first and supine:
        label, text = "head_first_supine", "Head First Supine"
    elif head_first and not supine:
        label, text = "head_first_prone", "Head First Prone"
    elif not head_first and supine:
        label, text = "feet_first_supine", "Feet First Supine"
    else:
        label, text = "feet_first_prone", "Feet First Prone"

    # ---- Confidence & why ----
    conf_terms = [v for v in [nc, le, re, scL, scR, lc, rc] if v > 0]
    mean_c = float(np.mean(conf_terms)) if conf_terms else 0.0
    conf = float(round(0.4 + 0.3 * min(1.0, mean_c), 3))

    why = (
        f"yolo_rules nose={nc:.2f} eyes={(le+re)/2:.2f} "
        f"shoulders={(scL+scR)/2:.2f} ankles={(lc+rc)/2:.2f}"
    )

    return {"label": label, "position": text, "confidence": conf, "why": why, "res": res}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIXED: ankle override sign now matches segmentation & YOLO
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def _override_head_first_with_ankles(seg, yolo_res):
    """
    Use YOLO ankle Y-positions to decide head/feet robustly.
    Returns (head_first_bool_or_None, used_override_bool).
    """
    try:
        rlist = yolo_res.get("res")
        if not rlist:
            return (None, False)
        r0 = rlist[0]
        if r0.keypoints is None or r0.keypoints.xy is None or len(r0.keypoints.xy) == 0:
            return (None, False)

        kxy = r0.keypoints.xy[0].cpu().numpy()
        kcf = r0.keypoints.conf[0].cpu().numpy() if r0.keypoints.conf is not None else None

        def xy(i):
            try:
                x, y = float(kxy[i, 0]), float(kxy[i, 1])
                if math.isnan(x) or math.isnan(y):
                    return None, None
                return x, y
            except Exception:
                return None, None

        def conf(i):
            try:
                if kcf is None:
                    return 0.0
                v = float(kcf[i])
                return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
            except Exception:
                return 0.0

        LANK, RANK = 15, 16
        _, ly = xy(LANK)
        _, ry = xy(RANK)
        lc, rc = conf(LANK), conf(RANK)

        if ly is None and ry is None:
            return (None, False)

        mean_c = (lc + rc) / 2.0
        if mean_c < 0.50:
            return (None, False)

        feet_y = np.nanmean([v for v in [ly, ry] if v is not None])
        ORIENT = ORIENT_HEAD_TOWARD

        # Prefer segmentation endpoints if available — **fixed sign**:
        if seg.get("head_end") is not None and seg.get("feet_end") is not None:
            he_y = float(seg["head_end"][1])
            fe_y = float(seg["feet_end"][1])
            if ORIENT == "TOP":
                head_first = (he_y < fe_y)
            else:  # BOTTOM
                head_first = (he_y > fe_y)
        else:
            # derive image height
            H = None
            m = seg.get("mask")
            if isinstance(m, np.ndarray):
                H = m.shape[0]
            if H is None:
                try:
                    H = int(r0.orig_shape[0]) if hasattr(r0, "orig_shape") else None
                except Exception:
                    H = None
            if H is None:
                return (None, False)

            # don't override if feet near midline
            if abs(feet_y - (H * 0.5)) < (0.05 * H):
                return (None, False)

            if ORIENT == "TOP":
                head_first = (feet_y >= (H * 0.5))
            else:  # BOTTOM
                head_first = (feet_y <= (H * 0.5))

        return (head_first, True)
    except Exception:
        return (None, False)

def _is_decubitus(lbl: str) -> bool:
    return "decubitus" in (lbl or "")

# =========================
# ------- Fusion  ---------
# =========================
def decide_hybrid(img, want_debug=False):
    """
    Fuse Segmentation (PCA + face cue) with YOLOv8-Pose.
    Applies ankle override, supine safeguards, and head/feet debouncing.
    """
    seg = classify_segmentation(img, want_debug=want_debug)
    yolo_res = yolo_pose_classify(img)

    best = seg if seg.get("confidence", 0.0) >= yolo_res.get("confidence", 0.0) else yolo_res

    seg_dbg   = seg.get("debug") or {}
    axis_vert = seg.get("axis_vert", seg_dbg.get("axis_vert"))
    face_vis  = seg.get("face_vis",  seg_dbg.get("face_vis", 0.0))
    is_decub  = ("decubitus" in (seg.get("label") or ""))

    if is_decub and axis_vert is not None and axis_vert < DECUB_AXIS_DEG_STRICT:
        seg = dict(seg)
        seg["confidence"] *= 0.6
        if best is seg:
            best = seg

    hf_new, used_override = _override_head_first_with_ankles(seg, yolo_res)

    def _flip(label: str, pos: str, head_first_now: bool):
        if head_first_now and label.startswith("feet_first_"):
            return label.replace("feet_first_", "head_first_"), pos.replace("Feet First", "Head First")
        if (not head_first_now) and label.startswith("head_first_"):
            return label.replace("head_first_", "feet_first_"), pos.replace("Head First", "Feet First")
        return label, pos

    if used_override and hf_new is not None:
        b_lab, b_pos = _flip(best.get("label",""), best.get("position",""), hf_new)
        if (b_lab, b_pos) != (best.get("label"), best.get("position")):
            best = dict(best)
            best["label"], best["position"] = b_lab, b_pos
            best["why"] = (best.get("why") or "") + " | applied_ankle_override"
        s_lab, s_pos = _flip(seg.get("label",""), seg.get("position",""), hf_new)
        if (s_lab, s_pos) != (seg.get("label"), seg.get("position")):
            seg = dict(seg)
            seg["label"], seg["position"] = s_lab, s_pos

    if is_decub and "decubitus" not in (yolo_res.get("label") or ""):
        if yolo_res.get("confidence", 0.0) >= 0.55 and seg.get("confidence", 0.0) <= 0.70:
            best = yolo_res

    if "decubitus" not in (seg.get("label") or "") and "decubitus" not in (yolo_res.get("label") or ""):
        if seg.get("label") == yolo_res.get("label"):
            best = dict(best)
            best["confidence"] = float(min(1.0, 0.5 * seg["confidence"] + 0.5 * yolo_res["confidence"]))

    def _nose_shoulders_ankles(yolo_res_dict):
        try:
            r0 = yolo_res_dict.get("res")[0]
            kcf = r0.keypoints.conf[0].cpu().numpy()
            def g(i):
                try:
                    v = float(kcf[i])
                    return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))
                except Exception:
                    return 0.0
            nose = g(0)
            shoulders = (g(5) + g(6)) / 2.0
            ankles = (g(15) + g(16)) / 2.0
            return nose, shoulders, ankles
        except Exception:
            return 0.0, 0.0, 0.0

    nose_c, shoulders_c, mean_ank = _nose_shoulders_ankles(yolo_res)

    if best.get("label") in ("head_first_prone", "feet_first_prone"):
        flipped = False
        if face_vis >= 0.5 or nose_c >= 0.20:
            best = dict(best)
            best["label"]      = best["label"].replace("prone", "supine")
            best["position"]   = best["position"].replace("Prone", "Supine")
            best["confidence"] = float(max(best.get("confidence", 0.55), 0.80))
            best["why"]        = (best.get("why") or "") + " | supine_override(face/nose_detected)"
            flipped = True
        axis_ok = (axis_vert is None) or (axis_vert < 25.0)
        if not flipped and axis_ok and (mean_ank >= 0.40 or shoulders_c >= 0.25):
            best = dict(best)
            best["label"]      = best["label"].replace("prone", "supine")
            best["position"]   = best["position"].replace("Prone", "Supine")
            best["confidence"] = float(max(best.get("confidence", 0.55), 0.75))
            best["why"]        = (best.get("why") or "") + " | supine_override(straight_axis_or_no_seg+ankles/shoulders)"

    new_lab, tag = _apply_headfeet_sticky(best["label"])
    if tag:
        best = dict(best)
        if new_lab.startswith("head_first_"):
            best["position"] = best["position"].replace("Feet First", "Head First")
        else:
            best["position"] = best["position"].replace("Head First", "Feet First")
        best["label"] = new_lab
        best["why"] = (best.get("why") or "") + " | " + tag

    return best, seg, yolo_res

# =========================
# ------- API -------------
# =========================
def _draw_title(vis, text: str):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 12
    cv2.rectangle(vis, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), (0, 0, 0), -1)
    cv2.putText(vis, text, (10 + pad, 10 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def decode_image_exif_aware(data: bytes, content_type: Optional[str]) -> np.ndarray | None:
    """Decode bytes -> BGR np.array, applying EXIF orientation for JPEGs."""
    try:
        ct = (content_type or "").lower()
        if ct in ("image/jpeg", "image/jpg"):
            pil = Image.open(BytesIO(data))
            pil = ImageOps.exif_transpose(pil)
            rgb = np.array(pil)
            if rgb.ndim == 3 and rgb.shape[2] == 3:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if rgb.ndim == 2:
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.get("/health")
def health():
    return {"ok": True, "service": APP_TITLE, "orient_head_toward": ORIENT_HEAD_TOWARD, "yolo_model": YOLO_MODEL}

# @app.post("/analyze", response_model=AnalyzeOut)
# async def analyze(
#     image: UploadFile = File(...),
#     debug: int = Query(0),
#     orient: str | None = Query(None, description="Override head side: TOP or BOTTOM")
# ):
#     try:
#         if not (image.content_type or "").startswith("image/"):
#             return JSONResponse({"error": "unsupported_media_type"}, status_code=415)

#         data = await image.read()
#         if len(data) > MAX_IMAGE_BYTES:
#             return JSONResponse({"error":"payload_too_large"}, status_code=413)

#         img = decode_image_exif_aware(data, image.content_type)
#         if img is None:
#             return JSONResponse({"error":"invalid_image"}, status_code=400)

#         # ---- optional per-request orientation override ----
#         global ORIENT_HEAD_TOWARD
#         requested_orient = _norm_orient(orient)
#         prev_orient = ORIENT_HEAD_TOWARD
#         used_orient = prev_orient

#         try:
#             if requested_orient:
#                 # lock is optional but recommended if you expect concurrent calls
#                 with ORIENT_LOCK:
#                     ORIENT_HEAD_TOWARD = requested_orient
#                     used_orient = ORIENT_HEAD_TOWARD
#                     print(f"[API] /analyze override orient -> {used_orient}", flush=True)
#                     best, seg, yolo_res = decide_hybrid(img, want_debug=bool(debug))
#             else:
#                 best, seg, yolo_res = decide_hybrid(img, want_debug=bool(debug))
#         finally:
#             # restore previous global if we changed it
#             if requested_orient:
#                 with ORIENT_LOCK:
#                     ORIENT_HEAD_TOWARD = prev_orient

#         dbg = None
#         if bool(debug):
#             dbg = {
#                 "seg_conf":  seg.get("debug", {}).get("seg_conf") if seg.get("debug") else None,
#                 "axis_vert": seg.get("debug", {}).get("axis_vert") if seg.get("debug") else None,
#                 "face_vis":  seg.get("debug", {}).get("face_vis")  if seg.get("debug") else None,
#                 "lr_thresh": seg.get("debug", {}).get("lr_thresh") if seg.get("debug") else None,
#                 "why_seg": seg.get("why"),
#                 "why_yolo": yolo_res.get("why") if yolo_res else None,
#                 "orient_used": used_orient,
#                 "orient_requested": requested_orient or None,
#             }

#         out = {
#             "position":   best["position"],
#             "label":      best["label"],
#             "confidence": float(best["confidence"]),
#             "why":        best["why"],
#             **({"debug": dbg} if dbg else {}),
#         }
#         return JSONResponse(content=jsonable_encoder(out), headers={"Cache-Control": "no-store"})
#     except Exception as e:
#         return JSONResponse({"error": f"server_error: {type(e).__name__}: {str(e)}"}, status_code=500)
    
@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(
    image: UploadFile = File(...),
    debug: int = Query(0),
    orient: str | None = Query(None, description="Override head side: TOP or BOTTOM")
):
    try:
        if not (image.content_type or "").startswith("image/"):
            return JSONResponse({"error": "unsupported_media_type"}, status_code=415)

        data = await image.read()
        if len(data) > MAX_IMAGE_BYTES:
            return JSONResponse({"error":"payload_too_large"}, status_code=413)

        img = decode_image_exif_aware(data, image.content_type)
        if img is None:
            return JSONResponse({"error":"invalid_image"}, status_code=400)

        # ---- optional per-request orientation override ----
        global ORIENT_HEAD_TOWARD
        requested_orient = _norm_orient(orient)
        prev_orient = ORIENT_HEAD_TOWARD
        used_orient = prev_orient

        try:
            if requested_orient:
                with ORIENT_LOCK:
                    ORIENT_HEAD_TOWARD = requested_orient
                    used_orient = ORIENT_HEAD_TOWARD
                    logging.info(f"[API] /analyze override orient -> {used_orient}")
                    best, seg, yolo_res = decide_hybrid(img, want_debug=bool(debug))
            else:
                best, seg, yolo_res = decide_hybrid(img, want_debug=bool(debug))
        finally:
            if requested_orient:
                with ORIENT_LOCK:
                    ORIENT_HEAD_TOWARD = prev_orient

        dbg = None
        if bool(debug):
            dbg = {
                "seg_conf":  seg.get("debug", {}).get("seg_conf") if seg.get("debug") else None,
                "axis_vert": seg.get("debug", {}).get("axis_vert") if seg.get("debug") else None,
                "face_vis":  seg.get("debug", {}).get("face_vis")  if seg.get("debug") else None,
                "lr_thresh": seg.get("debug", {}).get("lr_thresh") if seg.get("debug") else None,
                "why_seg": seg.get("why"),
                "why_yolo": yolo_res.get("why") if yolo_res else None,
                "orient_used": used_orient,
                "orient_requested": requested_orient or None,
            }

        out = {
            "position":   best["position"],
            "label":      best["label"],
            "confidence": float(best["confidence"]),
            "why":        best["why"],
            **({"debug": dbg} if dbg else {}),
        }

        # --- log the full output to backend console ---
        logging.info(f"[API] /analyze result: {jsonable_encoder(out)}")

        return JSONResponse(content=jsonable_encoder(out), headers={"Cache-Control": "no-store"})

    except Exception as e:
        logging.error(f"[API] /analyze server_error: {type(e).__name__}: {str(e)}")
        return JSONResponse({"error": f"server_error: {type(e).__name__}: {str(e)}"}, status_code=500)


@app.post("/overlay")
async def overlay(image: UploadFile = File(...)):
    try:
        data = await image.read()
        img = decode_image_exif_aware(data, image.content_type)
        if img is None:
            return JSONResponse({"error": "invalid_image"}, status_code=400)

        best, seg, yolo_res = decide_hybrid(img, want_debug=False)
        vis = draw_overlay(seg.get("img_pp", img), seg_result=seg, yolo_result=yolo_res)

        label_txt = f"{best['position']} ({int(best['confidence']*100)}%)"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        pad = 12
        cv2.rectangle(vis, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), (0, 0, 0), -1)
        cv2.putText(vis, label_txt, (10 + pad, 10 + th + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        ok, png = cv2.imencode(".png", vis)
        if not ok:
            return JSONResponse({"error": "encode_failed"}, status_code=500)

        resp = Response(content=png.tobytes(), media_type="image/png")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp
    except Exception as e:
        return JSONResponse({"error": f"server_error: {type(e).__name__}: {str(e)}"}, status_code=500)

# ================== Visualization helpers ==================
def draw_overlay(img_bgr, seg_result=None, yolo_result=None):
    img = img_bgr.copy()
    if seg_result and isinstance(seg_result.get("mask"), np.ndarray):
        mask = seg_result["mask"]
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0,220,80), 2)
        he, fe = seg_result.get("head_end"), seg_result.get("feet_end")
        if he is not None and fe is not None:
            cv2.circle(img, (int(he[0]), int(he[1])), 6, (0,255,255), -1)
            cv2.circle(img, (int(fe[0]), int(fe[1])), 6, (255,200,0), -1)
            cv2.line(img, (int(he[0]), int(he[1])), (int(fe[0]), int(fe[1])), (255,255,0), 2)
        face = seg_result.get("face")
        if face is not None:
            x,y,w,h = face
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,128,255), 2)

    if DRAW_SKELETON and yolo_result and yolo_result.get("res"):
        r0 = yolo_result["res"][0]
        if r0.keypoints is not None and r0.keypoints.xy is not None and len(r0.keypoints.xy) > 0:
            kxy = r0.keypoints.xy[0].cpu().numpy()
            for a,b in COCO_LINES:
                if a < len(kxy) and b < len(kxy):
                    xa,ya = kxy[a]; xb,yb = kxy[b]
                    if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                        cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0,255,0), 2)
            for (x,y) in kxy:
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(img, (int(x),int(y)), 3, (0,200,255), -1)
    return img

# ================ Optional webcam demo ================
_cam = None
_cam_lock = threading.Lock()
_live_label = {"position":"—","label":"unknown","confidence":0.0,"why":"init"}
_frame_idx = 0

def get_cam():
    global _cam
    with _cam_lock:
        if _cam is None:
            _cam = cv2.VideoCapture(0)
            _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return _cam

def gen_live_mjpeg():
    global _frame_idx, _live_label
    cap = get_cam()
    while True:
        with _cam_lock:
            ok = cap.isOpened()
        if not ok: break
        ok, frame = cap.read()
        if not ok: break

        if _frame_idx % PROCESS_EVERY_N == 0:
            best, seg, yolo_res = decide_hybrid(frame, want_debug=False)
            _live_label = {
                "position": best["position"], "label": best["label"],
                "confidence": float(best["confidence"]), "why": best["why"]
            }
            vis_src = seg.get("img_pp", frame)
            vis = draw_overlay(vis_src, seg_result=seg, yolo_result=yolo_res)
        else:
            vis = frame.copy()
            txt = f"{_live_label['position']} ({int(_live_label['confidence']*100)}%)"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(vis, (10,10), (10+tw+24, 10+th+24), (0,0,0), -1)
            cv2.putText(vis, txt, (22, 10+th+16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        _frame_idx += 1
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.get("/live_mjpeg")
def live_mjpeg():
    get_cam()
    return StreamingResponse(gen_live_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/live_status")
def live_status():
    return JSONResponse(content=jsonable_encoder(_live_label))

@app.on_event("shutdown")
def _shutdown():
    global _cam
    with _cam_lock:
        try:
            if _cam is not None:
                _cam.release()
        finally:
            _cam = None
