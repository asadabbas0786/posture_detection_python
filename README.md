Hybrid Patient Positioning – README

FastAPI service that classifies a patient’s orientation in an image using a hybrid of MediaPipe (segmentation + face) and YOLOv8-Pose (keypoints). The service returns a single verdict with confidence and “why” explanations.

🔭 What it detects

Plain postures

head_first_supine

head_first_prone

feet_first_supine

feet_first_prone

Side-lying (decubitus)

head_first_decubitus_left | head_first_decubitus_right

feet_first_decubitus_left | feet_first_decubitus_right

Fallback: unknown

🧠 How it works (two experts + fusion)

Segmentation branch (MediaPipe)

Segments the person, keeps the largest component.

Runs PCA on the mask to get a body axis and endpoints (proxy head/feet).

Detects face (HAAR → MediaPipe FaceDetector fallback).

Infers head/feet (face-closest endpoint = head) and supine iff face is visible.

Computes side-lying by left/right pixel imbalance; builds a blended confidence.

Pose branch (YOLOv8-Pose)

Reads nose, shoulders, ankles keypoints (+ confidences).

Head/feet: nose vs. ankles along Y; fallback to ankle vs image midline when nose is missing.

Supine: nose visible or any shoulder confident.

Confidence = base + fraction of mean keypoint conf; includes a “why” string.

Fusion layer

Picks the stronger branch, then applies rules:

Ankle override: if ankle evidence is good, force head/feet direction (robust even when segmentation fails).

Down-weight decubitus if the axis is very straight.

If seg says decubitus but YOLO confidently says plain posture, prefer YOLO.

If both plain postures agree, blend confidences for stability.

Supine safeguard: if result is prone but face/nose is visible (or axis straight + ankle/shoulder support), flip to supine.

⚙️ Configuration

Environment variables (read at startup):

Var	Default	Meaning
ORIENT_HEAD_TOWARD	TOP	Where the head is expected in the frame: TOP (smaller y) or BOTTOM (larger y). Keep this consistent across deployments.
YOLO_MODEL	yolov8s-pose.pt	Path to YOLOv8-Pose weights.
IMGSZ	1536	Inference image size for YOLOv8-Pose.
CORS_ALLOW_ORIGINS	http://localhost:3000,http://127.0.0.1:3000	Comma-separated origins for the browser frontend (LMS, etc.).
MAX_IMAGE_BYTES	10485760	Max upload size (bytes).
DECUB_AXIS_DEG_STRICT	25.0	Angle tolerance for “very straight” axis (used to soften decubitus).
DECUB_LR_THRESH_STRICT	0.60	L/R imbalance threshold when axis is straight.
DECUB_LR_THRESH_RELAX	0.12	L/R threshold when axis isn’t straight.
DRAW_SKELETON	1	Draw YOLO skeleton in /overlay.
PROCESS_EVERY_N	2	Frame sampling for the webcam demo.
SEGMENTER_MODEL	models/selfie_segmenter.tflite	MediaPipe segmenter model path.
FACEDET_MODEL	models/blaze_face_short_range.tflite	MediaPipe face detector model path.

Important: Call GET /health and verify it prints
"orient_head_toward": "TOP" (or BOTTOM) in every environment (CLI, LMS, server).

📦 Requirements

Python 3.10+

FastAPI, Uvicorn

Ultralytics (YOLOv8-Pose)

MediaPipe Tasks (tflite)

OpenCV (headless is fine for server)

NumPy, Pydantic, Starlette, etc.

Install from your pinned requirements.txt, e.g.:

python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt


On CPU-only Linux boxes, you can preinstall CPU wheels for PyTorch (if your YOLO build needs it):

pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

▶️ Run
# Linux/macOS
export ORIENT_HEAD_TOWARD=TOP      # or BOTTOM
uvicorn app:app --host 0.0.0.0 --port 5001

# Windows PowerShell
$env:ORIENT_HEAD_TOWARD = "TOP"    # or "BOTTOM"
uvicorn app:app --host 0.0.0.0 --port 5001


Check:

curl -s http://127.0.0.1:5001/health
# {"ok":true,"service":"Hybrid Patient Positioning","orient_head_toward":"TOP","yolo_model":"yolov8s-pose.pt"}

🧪 API
GET /health

Returns service info and current orientation rule.

POST /analyze?debug=1

Request: multipart form with image (jpeg/png).

Response:

{
  "position": "Head First Supine",
  "label": "head_first_supine",
  "confidence": 0.75,
  "why": "yolo_fallback(nose_vs_ankles) ... | applied_ankle_override | supine_override(...)",
  "debug": {
    "seg_conf": 0.0,
    "axis_vert": null,
    "face_vis": null,
    "lr_thresh": null,
    "why_seg": "segmentation_failed",
    "why_yolo": "yolo_fallback(...)"
  }
}


Linux/macOS curl

curl -s -X POST "http://127.0.0.1:5001/analyze?debug=1" \
  -H "accept: application/json" \
  -F "image=@pose/7.png;type=image/png"


Windows PowerShell (use curl.exe)

$img = "C:\path\to\pose\7.png"
curl.exe -s -X POST "http://127.0.0.1:5001/analyze?debug=1" `
  -H "accept: application/json" `
  -F "image=@$img;type=image/png"

POST /overlay

Returns PNG with contours, PCA line (head/feet), face box, and YOLO skeleton.

curl -s -X POST "http://127.0.0.1:5001/overlay" \
  -F "image=@pose/7.png;type=image/png" -o overlay.png

Optional webcam demo

GET /live_mjpeg – Motion-JPEG stream with overlays.

GET /live_status – Latest predicted label.

🧩 Image handling notes

Preprocess: optional shrink, bilateral denoise, CLAHE for more stable keypoints.

EXIF rotation: if you expect rotated JPEGs from phones, ensure decoding applies EXIF orientation consistently across your clients. The code includes a decode_image_exif_aware(...) helper you can use inside /analyze instead of raw cv2.imdecode.

🔁 Why LMS and CLI can differ

Different orientation env (ORIENT_HEAD_TOWARD not set or mismatched).

EXIF rotation applied by one path but not the other.

Resizing/cropping done by the LMS before upload.

Model versions or cache differences (ensure same weights and versions).

Checklist

Call /health from LMS; confirm "orient_head_toward":"TOP" (or your choice).

Upload exactly the same file (byte-identical).

If LMS uses JPEGs from mobile, enable EXIF-aware decoding.

Compare returned why fields (nose/shoulder/ankle confidences) to see which cue changed.

📂 Suggested project layout
.
├─ app.py
├─ requirements.txt
├─ yolov8s-pose.pt
├─ models/
│  ├─ selfie_segmenter.tflite
│  └─ blaze_face_short_range.tflite
└─ pose/            # sample images

🔧 Useful commands
# Kill a leftover server on port 5001 (Linux)
ss -ltnp | grep :5001
kill -9 <pid>

# Quick batch test (Linux)
for f in pose/*.jpeg pose/*.png; do
  echo "== $f ==";
  curl -s -X POST "http://127.0.0.1:5001/analyze?debug=1" \
    -H "accept: application/json" \
    -F "image=@$f;type=image/${f##*.}";
  echo;
done

📜 Output fields recap

position – human-readable label (e.g., "Head First Supine").

label – machine label (snake_case).

confidence – 0–1 score (blended/overridden as needed).

why – reasoning summary (what cues were used/overridden).

debug – optional extra cues when ?debug=1.

🙏 Acknowledgements

[Ultralytics YOLOv8-Pose]

[MediaPipe Tasks: Selfie Segmentation, Blaze Face]

OpenCV, FastAPI, Starlette, Pydantic
