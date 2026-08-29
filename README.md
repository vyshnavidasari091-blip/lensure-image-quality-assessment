# Lensure — AI-Powered Image Quality & Defect Detection

Lensure is a full-stack app that takes an uploaded image, runs it through a set of interpretable image-quality features plus a small hybrid deep-learning model, and returns a structured quality assessment: an overall score, a label (ACCEPTABLE / DEGRADED / DEFECTIVE), and a per-issue breakdown with severity and confidence. Results are stored and can be browsed later from a history view.

No external AI or vision APIs are used anywhere in the pipeline, and no API keys are needed.

```
┌────────────┐    upload    ┌──────────────┐   features + CNN   ┌─────────────┐
│  Frontend  │ ───────────► │  FastAPI     │ ──────────────────►│ Hybrid IQA  │
│ (HTML/JS)  │ ◄─────────── │  backend     │ ◄────────────────── │  model      │
└────────────┘   JSON       └──────┬───────┘     score/issues    └─────────────┘
                                    │
                                    ▼
                              SQLite (history)
```

## 1. Setup and running locally

This project runs directly with Python — there's no Docker setup involved. Clone the repo, then:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model once — this writes app/ml/weights/iqa_head.pt and metrics.json
python -m app.ml.train

# Start the API (it also serves the frontend from the same origin)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in a browser.

If you skip the training step, the API will still run — it just falls back to a rule-based scorer (you'll see `model_version: "rule-based-v1(no-weights)"` in the responses). Run `train.py` once beforehand to get the full hybrid model and proper grading credit.

## 2. Approach

**What it detects:** blur, underexposure, overexposure, noise, corruption/severe degradation, and a general "potential visual defect" catch-all.

**Method — hybrid, per section 3 of the brief:**

1. **Classical, engineered features** (`backend/app/ml/features.py`): sharpness via Laplacian variance, brightness/exposure histograms, contrast, a single-image noise-sigma estimate (Immerkaer's 1996 method — convolving with a Laplacian-like mask that cancels out flat/edge structure so what's left is mostly noise), the Hasler–Süsstrunk colorfulness metric, grayscale entropy, and Canny edge density. These are the same kind of natural-scene statistics used across no-reference IQA work (BRISQUE and similar), and on their own they're already enough to build a transparent, rule-based detector.

2. **A lightweight CNN**, trained from scratch and fused with those classical features (`backend/app/ml/model.py`): four small conv/BN/ReLU blocks, a global pool, concatenated with the 10 classical features, feeding into two heads — a 6-way distortion classifier (clean / blur / underexposed / overexposed / noise / corrupt) and a 0–100 quality-score regressor.

3. **Fusion at inference** (`backend/app/ml/infer.py`): the rule-based detector and the CNN both vote on issues. The final quality score is a weighted blend of the rule-based penalty score and the CNN's regression output, and the label is thresholded from that score, with any corruption signal forcing DEFECTIVE.

### Why a from-scratch CNN instead of transfer learning

The brief allows either approach. I went with a small CNN trained from scratch mainly because it needs no pretrained-weight download, so the whole thing stays offline and reproducible, and it trains in a couple of minutes on CPU. Since the classical features already carry a lot of signal for this task, the CNN mostly just needs to pick up complementary texture and context cues rather than learn a full representation from nothing. Swapping in a MobileNetV2 or ResNet18 backbone (`torchvision.models`, `weights="IMAGENET1K_V1"`) would be a small change to `model.py` if internet access to `download.pytorch.org` is available in the deployment environment.

## 3. Training data — synthetic distortions

Real IQA datasets like LIVE, TID2013, and KADID-10k are built the same way: start from pristine images and apply controlled degradations at multiple severities, so the model learns to recognize each distortion type without ever seeing a reference image at inference time.

Rather than downloading a dataset, this project generates its own pristine images procedurally (`backend/app/ml/synth_data.py`) — smooth low-frequency gradients (sky/wall-like regions), hard-edged shapes, fine speckle texture, and thin high-contrast lines. This keeps the pipeline fully offline and reproducible. Five controlled degradations are then applied at random severities: Gaussian blur, multiplicative under-/over-exposure, additive Gaussian noise, and simulated corruption (random block replacement plus heavy JPEG re-compression).

**If you want to improve generalization to real photos:** drop a folder of your own clean images into `backend/app/ml/pristine/` and swap out `make_pristine_image()` in `synth_data.py` for a loader that samples from that folder. The degradation, training, and evaluation code doesn't care where the clean images come from.

Thresholds for the rule-based detector (`SHARPNESS_BLUR_THRESH`, `NOISE_SIGMA_THRESH`, etc. in `infer.py`) were set empirically, by looking at the feature distributions for each synthetic class and picking cut-points that separate them.

## 4. Evaluation 

Run `python -m app.ml.train` to reproduce this. It evaluates on a held-out validation set of 300 synthetic images not seen during training.

| Metric    | Result |
|------------------  |
| Accuracy  | 98.33% |
| Precision | 98.39% |
| Recall    | 98.11% |
| F1 Score  | 98.20% |
| Quality Score MAE | 7.39 |

These numbers are from the synthetic validation set only — they shouldn't be read as equivalent to real-world photo accuracy. The confusion matrix and full metrics are saved automatically to `backend/app/ml/weights/metrics.json`.

### Failure cases and limitations

The validation set is procedurally generated, and real photographs bring in combinations of blur, noise, exposure issues, compression, camera-specific artifacts, and scene content that don't show up in synthetic training data. A few specific weak spots:

- Images with several problems at once (blur + noise + poor lighting together) are harder to classify cleanly.
- Blur, noise, and corruption can look similar at low severity, so the model sometimes confuses these classes.
- Different phones and cameras have their own exposure, compression, and sensor-noise characteristics that the synthetic data doesn't capture.
- Faces, text, very dark scenes, unusual textures, and highly detailed photos can behave differently than the synthetic images.
- The quality score is an indicative measure, not an absolute professional rating — its thresholds were calibrated on synthetic data.

So the 98.33% accuracy figure should be read as synthetic validation performance; real-world performance is likely lower.

## 5. Explainability 

Every response includes the raw stats behind the decision — sharpness, noise_sigma, brightness_mean, overexposed/underexposed fraction, entropy, edge_density, colorfulness — plus, when the CNN is loaded, its full per-class probability distribution (`cnn_class_probs`). That way a reviewer can see why an issue was flagged, not just that it was. Each issue also carries its own confidence score.

## 6. API reference 

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check + whether the trained model is loaded |
| POST | `/api/analyze` | Upload an image (`multipart/form-data`, field `file`) → structured result, persisted |
| GET | `/api/results?limit=&offset=` | Paginated list of past analyses |
| GET | `/api/results/{id}` | Full stored result for one analysis |
| GET | `/api/results/{id}/image` | The originally uploaded image |

Example:

```bash
curl -F "file=@photo.jpg;type=image/jpeg" http://localhost:8000/api/analyze
```

```json
{
  "id": 1,
  "filename": "photo.jpg",
  "quality_score": 54.1,
  "quality_label": "DEGRADED",
  "issues": [{"type": "blur", "severity": "high", "confidence": 0.97}],
  "stats": {"sharpness": 2.6, "noise_sigma": 0.28, "brightness_mean": 127.4, "...": "..."},
  "model_version": "hybrid-v1",
  "created_at": "2026-08-29T12:01:10.731384"
}
```

Invalid or unreadable files return 422 with a descriptive error (and are still logged with `quality_label: "INVALID"` for auditability). Oversized files return 413, wrong content types return 415.

## 7. Database 

SQLite by default, no setup needed, via SQLAlchemy. `backend/app/db/models.py` defines a single `analysis_results` table (id, filename, stored_path, quality_score, quality_label, issues (JSON), stats (JSON), model_version, error, created_at). Tables are created automatically on startup via `Base.metadata.create_all`. To switch to Postgres, just set `DATABASE_URL=postgresql://user:pass@host:5432/dbname` — no code changes required.

## 8. Frontend

Plain HTML/CSS/JS, no build step, served by FastAPI at `/` from `frontend/`. Supports drag-and-drop or click-to-upload, a dial-style display for the overall score, a per-issue list with severity and confidence, a stats panel, and a History tab backed by `GET /api/results`. Loading, success, and error states are all handled explicitly — see `frontend/app.js`.

## 9. Deployment notes 

This was run and tested locally with `uvicorn`, without Docker. The source code is pushed to this GitHub repository, which anyone can clone and run following the setup steps in section 1 above.

- `GET /health` acts as the service-check endpoint.
- Configurable via environment variables: `DATABASE_URL`, `UPLOAD_DIR`, `MODEL_PATH`, `MAX_UPLOAD_MB`.
- The SQLite database and uploaded images persist on disk between runs.
- Not deployed to a public URL for this submission — local setup only, which the brief allows ("local deployment acceptable").

## 10. Repository layout

```
backend/
  app/
    api/        # FastAPI routes + pydantic schemas
    core/       # settings (env vars)
    db/         # SQLAlchemy engine + ORM model
    ml/
      features.py     # classical feature extraction
      synth_data.py   # procedural pristine images + degradations
      model.py         # hybrid CNN definition
      train.py         # training + evaluation script
      infer.py         # inference pipeline used by the API
      weights/         # iqa_head.pt + metrics.json (generated)
    main.py     # app entrypoint
  requirements.txt
frontend/       # static HTML/CSS/JS
samples/        # example images across each quality condition
```

## 11. Sample images 

`samples/` has one procedurally-generated example per condition (`clean.jpg`, `blur.jpg`, `noise.jpg`, `overexposed.jpg`, `underexposed.jpg`, `corrupt.jpg`) for quick manual testing — upload any of them through the UI or via `curl` to see each issue type detected end-to-end.

## 12. Bonus items implemented

- Confidence scores per issue
- Full analysis history with a browsable list endpoint and UI tab
- Health/status endpoint
- Graceful degradation to a rule-based fallback if model weights are absent
