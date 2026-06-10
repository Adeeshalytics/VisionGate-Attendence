# VisionGate

**Face Recognition–Based Attendance System**
University of Ruhuna — Faculty of Engineering
Module: EE7204 / EC7205 — Digital Image Processing (Mini Project)

> *Team members:* _Add your team member names and registration numbers here._

---

## 1. Overview

VisionGate is an end-to-end attendance system that uses a single webcam to
detect, recognize and mark students present in a lecture or lab. The system
is designed for classroom-scale deployments (≤ ~100 enrolled students),
runs entirely on CPU, and stores attendance in a local SQLite database.

The pipeline favours well-understood, reproducible components:
**MediaPipe BlazeFace** for detection, **dlib ResNet 128-d embeddings**
(via `face_recognition`) for identification, **OpenCV LBPH** as a secondary
comparator, and **dlib 68-point landmarks** for an **Eye Aspect Ratio (EAR)**
blink-based liveness gate.

## 2. System architecture

```
                ┌──────────────────────────────┐
                │           Webcam             │
                └──────────────┬───────────────┘
                               │ raw BGR frames
                               ▼
                ┌──────────────────────────────┐
                │   preprocess.full_pipeline   │
                │  resize → denoise → CLAHE    │
                └──────────────┬───────────────┘
                               │ (color, gray-enhanced)
                               ▼
                ┌──────────────────────────────┐
                │      Face Detection          │
                │ MediaPipe BlazeFace (primary)│
                │   Haar Cascade (fallback)    │
                └──────────────┬───────────────┘
                               │ bounding boxes
                ┌──────────────┴────────────────┐
                ▼                               ▼
   ┌────────────────────────┐    ┌──────────────────────────────┐
   │  Enrollment (one-time) │    │   Recognition (live loop)    │
   │  ─────────────────     │    │  ─────────────────────────   │
   │ crop ROI (+20px pad)   │    │ crop ROI                     │
   │ augment_image × 7      │    │ face_recognition.encodings   │
   │ face_encodings()       │    │ face_distance vs gallery     │
   │ train LBPH             │    │ best_dist ≤ threshold?       │
   │ save encodings.pkl     │    │   ↓ yes                      │
   │ save lbph_model.yml    │    │ LivenessDetector (EAR blink) │
   │ register_student()     │    │   ↓ is_live?                 │
   └────────────┬───────────┘    │ mark_attendance()            │
                │                │ (duplicate window enforced)  │
                ▼                └──────────────┬───────────────┘
   data/enrolled_faces/<sid>/                   │
   data/encodings/encodings.pkl                 ▼
   data/encodings/lbph_model.yml   ┌──────────────────────────────┐
                                   │       SQLite (sqlite3)       │
                                   │  students | attendance       │
                                   └──────────────┬───────────────┘
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Streamlit Dashboard          │
                                   │ today | history | registry   │
                                   │ analytics                    │
                                   └──────────────────────────────┘
```

## 3. Requirements

- **Python:** 3.10 or newer (tested on 3.14 / Windows 11)
- **OS:** Windows / macOS / Linux
- **Hardware:** any USB or built-in webcam (640×480 minimum); discrete GPU
  is **not** required.
- **Disk:** ~250 MB free (mostly for the dlib landmark model and dependencies).

## 4. Installation

```bash
git clone <your-repo-url> visiongate
cd visiongate
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first time you launch the system, two model files are downloaded
automatically on demand:

- `models/blaze_face_short_range.tflite` (MediaPipe, ~230 KB)
- `models/shape_predictor_68_face_landmarks.dat` (dlib, ~99 MB)

## 5. Quick start

All workflows are reachable through `main.py`:

```bash
python main.py
```

| Option | What it does |
| :---:  | --- |
| 1 | **Enroll a new student** — prompts for name/ID, captures `ENROLLMENT_IMAGES` raw face crops, augments them 7× per crop, trains 128-d encodings + LBPH, persists to disk and DB. |
| 2 | **Start an attendance session** — opens the recognition loop with liveness gating. Press **Q** to end the session and auto-export today's CSV. |
| 3 | **Open the dashboard** — launches `streamlit run dashboard.py` on `http://localhost:8501` and opens your default browser. |
| 4 | **Export today's CSV** — writes `data/attendance/attendance_<YYYY-MM-DD>.csv`. |
| 5 | **Print today's summary** — formatted console table of today's attendance. |
| 6 | Exit. |

Individual modules can also be run directly:

```bash
python enroll.py            # enrollment only
python recognize.py         # recognition only
streamlit run dashboard.py  # dashboard only
pytest tests/ -v            # run the test suite
```

## 6. Configuration

All tunable parameters live in [config.py](config.py) and can be overridden
through a `.env` file in the project root (e.g. `RECOGNITION_THRESHOLD=0.42`).

| Constant | Default | Meaning |
| --- | :---: | --- |
| `CAMERA_INDEX` | `0` | Index passed to `cv2.VideoCapture`. |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `640 × 480` | Capture resolution. |
| `FRAME_SKIP` | `3` | Process every Nth frame in the recognition loop. |
| `ENROLLMENT_IMAGES` | `20` | Raw captures per enrollment (multiplied 7× by augmentation). |
| `RECOGNITION_THRESHOLD` | `0.45` | Maximum `face_recognition` distance for a positive match (lower = stricter). |
| `MIN_FACE_CONFIDENCE` | `0.75` | MediaPipe minimum detection confidence. |
| `MIN_FACE_SIZE` | `(60, 60)` | Smallest accepted bounding box. |
| `DUPLICATE_WINDOW_SECONDS` | `300` | Reject duplicate attendance within this window. |
| `EAR_THRESHOLD` | `0.25` | Eye Aspect Ratio below which a frame counts as "eye closed". |
| `EAR_CONSEC_FRAMES` | `2` | Consecutive sub-threshold frames required to count one blink. |
| `LOG_LEVEL` | `"INFO"` | Python logging level for `logs/visiongate.log`. |

## 7. Project layout

```
visiongate/
├── config.py            # paths, thresholds, .env overrides
├── database.py          # SQLite schema, attendance + duplicate window
├── preprocess.py        # resize, CLAHE, denoise, augmentation
├── utils.py             # logging, dir bootstrap, drawing helpers
├── enroll.py            # MediaPipe + Haar, dlib encodings, LBPH training
├── anti_spoof.py        # EAR blink-based liveness gate
├── recognize.py         # live recognition loop + attendance writing
├── dashboard.py         # Streamlit UI (4 tabs)
├── validate.py          # model validation / evaluation suite
├── evaluation.ipynb     # notebook that renders the evaluation inline
├── main.py              # CLI menu entry point
├── data/
│   ├── attendance.db    # SQLite database
│   ├── enrolled_faces/  # raw crops per student
│   ├── encodings/       # encodings.pkl + lbph_model.yml
│   ├── attendance/      # exported CSVs
│   ├── test_faces/      # (optional) drop-in test images per identity
│   └── validation/      # generated metrics + charts
├── models/              # cached MediaPipe + dlib model files
├── logs/                # rotating logs (visiongate.log)
└── tests/               # pytest suite (preprocess + database + validation)
```

## 8. Validation & evaluation

The recogniser is evaluated as a biometric system, not just demoed. Run:

```bash
python validate.py --source synthetic   # instant offline smoke test
python validate.py --source lfw          # real benchmark (LFW subset)
python validate.py --source auto         # LFW + your enrolled faces
```

or open `evaluation.ipynb` for the same results rendered inline.

**What is measured**

| Metric | Why it matters for an attendance system |
|---|---|
| Genuine vs. impostor distance distributions | Shows class separability of the embeddings |
| **FAR** (False Acceptance Rate) | A stranger being marked as a real student — the critical security failure |
| **FRR** (False Rejection Rate) | A real student being denied — the convenience failure |
| **ROC curve + EER** | Data-driven justification of `RECOGNITION_THRESHOLD` |
| Accuracy / Precision / Recall / F1 | Standard identification scorecard (macro-averaged) |
| Confusion matrix | Which identities get confused with whom |
| dlib ResNet vs. LBPH | Quantitative comparison of the two trained models |

Evaluation uses **stratified k-fold cross-validation** so the numbers are
not an artefact of one split. All charts and a `metrics_summary.json` are
written to `data/validation/`. To evaluate with your own classmates, drop
images into `data/test_faces/<name>/` and run `--source auto`.

The metric maths (FAR/FRR sweep, EER) is itself unit-tested in
`tests/test_validate.py`.

## 9. Known limitations

- **Heavy occlusion** (masks, large sunglasses, hands across the face) will
  degrade both detection and recognition accuracy.
- **Liveness detection requires adequate lighting**, frontal pose and an
  unobstructed view of both eyes — the blink-EAR heuristic is intentionally
  simple and is not a substitute for a dedicated anti-spoofing model.
- **CPU-only**: while no GPU is required, recognition throughput is bounded
  by CPU performance; on low-end laptops the effective frame rate drops to
  ~5–10 fps with `FRAME_SKIP=3`.
- **Single-camera, single-room**: the system has no notion of multiple
  cameras or sites.
- **No identity verification beyond face**: anyone who passes the liveness
  check and matches an enrolled embedding will be marked present.

## 10. Future improvements

- Replace the dlib ResNet embeddings with **InsightFace / ArcFace** to
  improve robustness to pose and illumination.
- Add **multi-camera support** for larger lecture halls (camera-side
  detection, server-side de-duplication).
- Add **mobile notifications** (e.g. via push or email) when a student is
  marked present.
- Add a **dedicated CNN-based anti-spoofing model** (printed-photo / replay
  attack detection) on top of the existing EAR gate.
- Add **role-based authentication** for the dashboard so only authorised
  staff can view rosters.

## 11. References

> _The bibliography for this mini-project is maintained in the project
> proposal document. Paste the proposal's reference list here verbatim
> before submitting — do not generate citations automatically._

<!--
Replace this block with the references from your proposal, in the same
formatting style your supervisor expects (IEEE / APA / etc.).
-->
