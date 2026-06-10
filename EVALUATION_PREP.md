# VisionGate — Evaluation Prep Plan
**Evaluation: tomorrow (2026-06-11) · University of Ruhuna · EE7204/EC7205**

This is your end-to-end checklist: improve → add faces → validate → make charts →
prep talking points → run the demo. Work top to bottom. Items are tagged
**[MUST]** (do before eval) or **[NICE]** (only if time).

---

## 0. Where you are right now (honest snapshot)

| Area | State |
|---|---|
| Core pipeline (preprocess, enroll, recognize, liveness) | ✅ Working |
| Backend tests | ✅ 18/18 passing |
| FastAPI backend (`api/`) | ✅ Works (deps now installed) |
| Next.js frontend (`frontend/`) | ✅ Builds & runs |
| Validation suite (`validate.py`) | ✅ Built |
| **Enrolled people** | ⚠️ **Only 1 (you)** |
| **Validation charts on disk** | ⚠️ **SYNTHETIC — not real, must regenerate** |
| Real benchmark numbers (LFW) | ❌ Not run yet |
| Live web demo with webcam | ❌ Never run end-to-end |
| README for web stack | ❌ Not written |

**The two things that will sink the eval if not fixed:** (1) only one face enrolled,
(2) the charts you'd show are fake placeholder data. Both are handled below.

---

## 1. THE KEY DECISION — what data do you use tomorrow?

You need data for **two different purposes**. Don't confuse them:

- **Live demo gallery** = who the camera will actually recognize on the day.
- **Validation dataset** = what your accuracy/charts are computed on.

### Recommended approach (most convincing, least risk)
1. **Live demo:** enroll **yourself + 2–4 classmates/teammates** tonight or 30 min
   before the eval. Real people the examiner can watch get recognized live.
2. **Validation/charts:** use **LFW** (a real public benchmark, many identities) +
   your enrolled faces. This gives credible FAR/FRR/accuracy numbers without
   needing 50 classmates.

### If you can't get other people
- Live demo: just you (green box on you, examiner's face shows as "Unknown" red box —
  that actually demonstrates the system rejecting strangers, which is a plus).
- Validation/charts: **LFW only** — still fully valid, it's the standard benchmark.

> **Clarify the meaning of "training":** dlib/face_recognition does **not** retrain a
> network when you add a face — it computes a 128-d embedding and stores it. "Adding a
> face" = **enrollment**, not training. The only thing that *trains* is the LBPH model
> (rebuilt automatically during enrollment). So you never wait for model training.

---

## 2. TONIGHT'S CHECKLIST (time-boxed, ~90 min)

### 2.1 [MUST] Regenerate REAL validation charts (~15 min, mostly download)
The charts on disk are synthetic. Replace them with real benchmark numbers:
```powershell
cd C:\Projects\VisionGate
python validate.py --source lfw --folds 5
```
- First run downloads ~200 MB LFW (one time). If the network is bad, fall back to:
  `python validate.py --source olivetti`
- Output charts land in `data\validation\` (overwrites the synthetic ones).
- Read the printed summary — note the EER, FAR/FRR @ 0.45, and dlib vs LBPH accuracy.
  **Write those numbers down for your slides.**

### 2.2 [MUST] Enroll real people for the live demo (~5 min each)
For each person (do this on the machine you'll demo on):
```powershell
python main.py        # choose 1, enter name + ID, look at camera until 20/20
```
or via the web UI later (Header → "Enroll student").
- Re-enroll yourself too if you want the denser, higher-quality gallery (the encoding
  fix from this week gives ~140 vs the old 106). Answer `y` to re-enroll.

### 2.3 [NICE] Add stricter validation methodology (~ask me, 10 min)
Right now identification uses **5-fold cross-validation** (proper train/test split),
but the verification EER threshold is picked on the same data (mildly optimistic).
I can add a **held-out calibration/evaluation split** so it's airtight. Ask me to do it.

### 2.4 [NICE] Sanity-run the full web stack once (~10 min) — see §6 for the runbook
Don't let tomorrow be the first time the web demo runs.

### 2.5 [MUST] Re-run tests so you can say "all tests pass" (~1 min)
```powershell
python -m pytest tests/ -q
```
Expect `18 passed`.

---

## 3. ADDING NEW FACES — exact how

### Option A — Live webcam (the built-in way)
```powershell
python main.py     # option 1
# or
python enroll.py --name "Jane Doe" --id "EG2021999"
```
Captures 20 frames → augments to ~140 → computes embeddings → updates
`encodings.pkl` + retrains LBPH → registers in DB → saves crops to
`data\enrolled_faces\<id>\`.

### Option B — From a folder of photos (NOT built yet)
If you want to enroll classmates from existing photos (no live webcam per person),
this script doesn't exist yet. **Ask me and I'll build `enroll_from_images.py`
tonight** — point it at `some_folder/<name>/*.jpg` and it enrolls everyone in one go.
Useful if teammates aren't physically present.

### How to verify a face was added
```powershell
python -c "import database as d; print([s['student_id'] for s in d.get_all_students()])"
```
or open the web Registry tab.

---

## 4. VALIDATION — commands + what every number means

### Run it
```powershell
python validate.py --source lfw          # real benchmark (recommended for slides)
python validate.py --source auto         # LFW + your enrolled faces
python validate.py --source enrolled     # only your enrolled people (weak: too similar)
python validate.py --source synthetic    # offline smoke test only — NOT for slides
```

### What each metric means (memorize these for Q&A)
| Metric | Plain meaning | Good value |
|---|---|---|
| **Genuine vs impostor distance** | same-person distances should be small, different-person large | clear gap |
| **FAR** (False Acceptance Rate) | stranger wrongly marked as an enrolled student | low (≈ EER) |
| **FRR** (False Rejection Rate) | real student wrongly denied | low (≈ EER) |
| **EER** (Equal Error Rate) | the point where FAR = FRR; single headline error number | lower = better |
| **EER threshold** | the distance at which EER occurs | should be ≈ your `RECOGNITION_THRESHOLD` 0.45 |
| **Accuracy / Precision / Recall / F1** | identification scorecard, macro-averaged | higher = better |
| **dlib vs LBPH** | your two models compared | dlib should win on LFW |

### The train/test split answer (you WILL be asked)
- **Identification (dlib + LBPH):** stratified **5-fold cross-validation** — train fold =
  gallery, test fold = probe, fully disjoint. Every image tested once; `accuracy_std`
  is the fold variance. Stronger than a single 80/20 split.
- **Verification (FAR/FRR/EER):** parameter-free pairwise distance protocol (no model
  trained), so no split needed — standard biometric practice. (Caveat: threshold is
  picked on the same data → see §2.3 if you want it airtight.)

---

## 5. CHARTS & GRAPHS FOR THE PRESENTATION

After §2.1, these files in `data\validation\` are slide-ready (PNG, 130 dpi):

| File | Put on slide | One-liner to say |
|---|---|---|
| `distance_distributions.png` | "Why it works" | "Genuine and impostor distances are clearly separable." |
| `roc_curve.png` | "Accuracy / ROC" | "ROC with EER marked — area under curve shows strong separation." |
| `far_frr.png` | "Threshold justification" | "FAR/FRR cross at ~0.44, validating our 0.45 threshold choice." |
| `confusion_dlib.png` | "Results — dlib" | "Confusion matrix from 5-fold CV; diagonal = correct." |
| `confusion_lbph.png` | "Baseline — LBPH" | "LBPH baseline for comparison." |
| `model_comparison.png` | "Model comparison" | "dlib ResNet beats LBPH on accuracy/precision/recall/F1." |
| `metrics_summary.json` | appendix / speaker notes | the raw numbers |

Plus screenshots to grab on the day:
- The **live recognition** window/feed with a green box + name (proof it works).
- The **dashboard** Today + Analytics tabs with real data.
- A **"Spoof"/"Blink to verify"** yellow box (proof of liveness).
- An **"Unknown"** red box on a non-enrolled person (proof it rejects strangers).

> Want me to also generate a **system architecture diagram** image for the slides?
> Ask and I'll produce one.

---

## 6. DEMO-DAY RUNBOOK (rehearse this exact sequence)

### Pre-flight (do 15 min before)
```powershell
cd C:\Projects\VisionGate
python -m pytest tests/ -q        # confirm green
```
- Close other apps using the webcam (Zoom/Teams/browser camera tabs).
- Make sure lighting is decent and you're 40–80 cm from the camera.

### Option 1 — Web stack (most impressive). Two terminals:
```powershell
# Terminal 1 (backend)
cd C:\Projects\VisionGate
python -m uvicorn api.main:app --port 8000

# Terminal 2 (frontend)
cd C:\Projects\VisionGate\frontend
npm run dev
```
Open **http://localhost:3000**. Demo flow:
1. Header → status dot should be **online**.
2. **Live session** tab → Start session → look at camera → **blink** → green box + your name.
3. Have a non-enrolled person appear → **red "Unknown"** box (great talking point).
4. **Today** tab → your attendance row appears + pie chart.
5. **Registry** tab → enrolled students.
6. **Analytics** tab → charts.
7. **History** tab → Download CSV.

### Option 2 — Simple/robust fallback (if web stack misbehaves)
```powershell
python main.py
# 1 = enroll, 2 = live recognition window, 3 = export CSV, 4 = today's summary
```
The native OpenCV window (option 2) is the most reliable recognition demo.

### Golden rule
**Have BOTH options ready.** If the browser/CORS/camera fights you, fall back to
`python main.py` → option 2 without missing a beat.

---

## 7. SLIDE DECK OUTLINE (≈10 slides)

1. **Title** — VisionGate, names, reg numbers, module.
2. **Problem** — manual attendance is slow/proxy-prone; face recognition automates it.
3. **System architecture** — the block diagram (webcam → preprocess → detect → encode →
   match → liveness → DB → dashboard).
4. **Pipeline details** — MediaPipe detection, dlib 128-d embeddings, LBPH baseline,
   CLAHE/denoise/augmentation.
5. **Anti-spoofing** — EAR blink liveness (show the formula + yellow "Spoof" screenshot).
6. **Validation methodology** — dataset (LFW), 5-fold CV, FAR/FRR/EER definitions.
7. **Results** — ROC + EER + confusion matrix + accuracy table (real numbers from §2.1).
8. **Model comparison** — dlib vs LBPH bar chart.
9. **Live demo** — switch to the running app.
10. **Limitations & future work** — occlusion, lighting, ArcFace upgrade, multi-camera.

---

## 8. Q&A PREP — likely questions + crisp answers

- **"How does recognition actually work?"** → Detect face (MediaPipe), crop, compute a
  128-d embedding with a dlib ResNet, compare to enrolled embeddings by Euclidean
  distance; if the nearest is within threshold 0.45, it's a match.
- **"Why this threshold?"** → Data-driven: our EER analysis shows FAR=FRR at ~0.44, so
  0.45 is justified, not guessed (show `far_frr.png`).
- **"Did you do a train/test split?"** → Yes — stratified 5-fold cross-validation for
  identification (gallery vs probe, disjoint). Verification EER is a parameter-free
  pairwise protocol.
- **"What's your accuracy / error rate?"** → [fill from §2.1 run] e.g. "EER ≈ X%,
  identification accuracy ≈ Y% across 5 folds."
- **"How do you stop photo spoofing?"** → EAR blink-liveness: we require a real blink
  (eye-aspect-ratio dips below 0.25 for ≥2 frames) before marking attendance.
- **"Why dlib and not a CNN/ArcFace?"** → dlib runs on CPU, is reproducible, and is
  strong enough for classroom scale; ArcFace is our stated future upgrade.
- **"Why also LBPH?"** → As a classical baseline to quantify how much the deep
  embedding helps (show the comparison chart).
- **"How does it scale / handle bad lighting / occlusion?"** → Designed for ≤~100
  students on CPU; CLAHE helps lighting; heavy occlusion is a known limitation.
- **"How is duplicate attendance prevented?"** → A 5-minute duplicate window per
  student in the DB layer.
- **"Where's the data stored?"** → SQLite (`students` + `attendance` tables), CSV export.

---

## 9. CONTINGENCIES / TROUBLESHOOTING

| Problem | Fix |
|---|---|
| Camera won't open | close other camera apps; try `CAMERA_INDEX=1` in a `.env` file |
| Keeps saying "Spoof / Blink to verify" | blink naturally a few times; improve lighting |
| Web status dot "offline" | the FastAPI server (terminal 1) isn't running |
| LFW download fails | use `python validate.py --source olivetti` |
| Frontend won't start | `cd frontend; npm install` then `npm run dev` |
| Everything is flaky | fall back to `python main.py` → option 2 (native window) |
| Recognition too strict/loose | adjust `RECOGNITION_THRESHOLD` in `config.py` (lower=stricter) |

---

## 10. PRIORITIZED TODO (if you only have limited time)

1. **[MUST]** `python validate.py --source lfw` → real charts + numbers (§2.1)
2. **[MUST]** Enroll yourself (re-enroll) + any classmates (§2.2)
3. **[MUST]** Rehearse the demo once, both web and CLI fallback (§6)
4. **[MUST]** Write your headline numbers into the slides (§5, §7)
5. **[NICE]** Ask me to add the held-out threshold split (§2.3)
6. **[NICE]** Ask me to build `enroll_from_images.py` if teammates aren't present (§3B)
7. **[NICE]** Ask me to generate an architecture diagram image (§5)
8. **[NICE]** README for the web stack

---

### Things only YOU can do (I can't): 
enroll real faces (needs your webcam), run the live camera demo, rehearse speaking.

### Things I can do for you right now (just say the word):
- Run the LFW validation and hand you real charts + numbers.
- Add the rigorous held-out threshold split.
- Build `enroll_from_images.py` for photo-based enrollment.
- Generate a system architecture diagram image.
- Write the web-stack section of the README.
