# Skin-Lesion AI Classification (HAM10000 / MobileNetV2)

An educational and research web application that classifies skin-lesion
images and explains each prediction with a Grad-CAM visualization.

> **Important:** This is an educational and research AI classification
> system, **not** a medical diagnosis and not a substitute for clinical
> assessment. The Grad-CAM visualization shows model influence only; it does
> not prove that a disease is present.

## Architecture

```
React frontend (static site)
        │  HTTPS (VITE_API_BASE_URL baked at build time)
        ▼
FastAPI backend  ──▶  Trained MobileNetV2 model (torch, CPU)
        │                       │
        │                       └─▶ Grad-CAM explanation (same model)
        ▼
PostgreSQL prediction history (psycopg 3)
```

| Component | Location | Notes |
| --- | --- | --- |
| React frontend | `frontend/` | Vite 5 + React 18; API URL injected at build time |
| FastAPI backend | `src/api.py` | Flat imports: start uvicorn from `src/` |
| Grad-CAM | `src/gradcam.py` | Pure torch/Pillow/numpy; runs on CPU |
| Data pipeline | `src/ham10000_data.py` | Training-time preprocessing (parity-verified) |
| DB layer | `src/db.py` | Env-only credentials; migrations auto-apply at startup |
| Trained model | `training_output/mobilenetv2_ham10000/` | `best_model.pt` (8.75 MB) + mapping + preprocessing JSON |
| Dataset | `dataset/` | ~8 GB, git-ignored, **never** deployed |

The model predicts 7 HAM10000 classes (akiec, bcc, bkl, df, mel, nv, vasc).

## Repository contents (deployment-relevant)

```
Skin-Disease-AI/
├── render.yaml                  # Render blueprint (DB + API + static site)
├── requirements.txt             # Python dependencies (torch 2.14.1, CPU-capable)
├── .env.example                 # Environment variable reference (no secrets)
├── .gitignore                   # Excludes dataset/, node_modules/, .env, caches
├── src/                         # FastAPI application
│   ├── api.py                   # App factory; `app = create_app()` at module level
│   ├── model_inference.py       # Model loading + prediction
│   ├── gradcam.py               # Grad-CAM generation
│   ├── db.py                    # PostgreSQL persistence + migrations
│   ├── init_database.py         # Optional local-DB bootstrap (not needed on Render)
│   └── ham10000_data.py         # Preprocessing shared with training
├── frontend/                    # React + Vite app
│   ├── package.json             # build: vite build
│   ├── .env.example             # VITE_API_BASE_URL reference
│   └── src/                     # App.jsx, main.jsx, styles.css
└── training_output/mobilenetv2_ham10000/
    ├── best_model.pt            # Trained weights — 8.75 MB (9,173,131 bytes)
    ├── class_label_mapping.json # Class id ↔ name mapping
    ├── inference_preprocessing.json
    └── (training reports/plots) # Small artifacts, safe to commit
```

The ~8 GB dataset (`dataset/`), `node_modules/`, build outputs, Python
caches, and all `.env` files are excluded by `.gitignore`. The trained model
(8.75 MB) **is** committed because the deployed API loads it at startup.

## Local development

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Optional: copy .env.example to .env and adjust (model paths, CORS,
# DATABASE_URL for a local PostgreSQL server).
copy .env.example .env

cd src
uvicorn api:app --host 127.0.0.1 --port 8000
```

The API serves `GET /health`, `POST /explain` (image + Grad-CAM), and the
`/predictions` history endpoints. Without `DATABASE_URL` the API runs fine
and history storage is disabled (`/health` reports `history_storage:
"not_configured"`).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173 (dev server)
```

For a production-style local build:

```bash
npm run build && npm run preview
```

When `VITE_API_BASE_URL` is unset, the frontend targets
`http://127.0.0.1:8000`.

### 3. Local PostgreSQL (optional)

```bash
set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5433/skin_lesion_ai
python src/init_database.py
```

Credentials come only from environment variables — never from the command
line, code, or committed files. When the API starts with `DATABASE_URL`
set, schema migrations apply automatically.

## Deploying to Render

### Prerequisites

- A GitHub (or GitLab) account; push this repository (the `.gitignore`
  guarantees the 8 GB dataset stays out).
- A Render account. The blueprint creates three resources:
  1. PostgreSQL database `skin-lesion-db`
  2. Web service `skin-lesion-ai-api` (Python 3.11)
  3. Static site `skin-lesion-ai-site` (Node 24, Vite build)

### Steps

1. **Push the repository** to GitHub. Verify `git status` shows no
   `dataset/`, `node_modules/`, `.env`, or cache files.

2. **Create the blueprint**: in Render, choose *New → Blueprint*, select the
   repository. Render reads `render.yaml` and proposes the three resources.

3. **Enter the two prompted values** (they reference each other's URLs):
   - `CORS_ORIGINS` on the API service: `https://skin-lesion-ai-site.onrender.com`
   - `VITE_API_BASE_URL` on the static site: `https://skin-lesion-ai-api.onrender.com`

   Use the exact URLs Render shows on the resources page if a random suffix
   was appended (name collisions do that).

4. **Apply**. Render builds and starts everything. The API applies database
   migrations automatically at startup — no manual DB step is needed.

5. **Verify**:
   - `https://<api-url>/health` → `{"status": "ok", "model_loaded": true, ...}`
   - Open the static site, upload a JPEG/PNG/WebP skin image, confirm a
     prediction, all-class probabilities, the Grad-CAM overlay, and that the
     entry appears in the prediction history.

6. **If URLs changed** (suffix appended): update `CORS_ORIGINS` on the API
   and `VITE_API_BASE_URL` on the static site in the Render dashboard. The
   site rebuilds automatically so the new API URL is baked in.

### Commands Render runs

| Resource | Build command | Start command |
| --- | --- | --- |
| `skin-lesion-ai-api` | `pip install -r requirements.txt` | `cd src && uvicorn api:app --host 0.0.0.0 --port $PORT` |
| `skin-lesion-ai-site` | `npm install && npm run build` | (static; serves `frontend/dist`) |

### Environment variables

| Variable | Where | Required | Purpose |
| --- | --- | --- | --- |
| `DATABASE_URL` | API (auto-wired by blueprint) | Yes | PostgreSQL connection; migrations run at startup |
| `CORS_ORIGINS` | API | Yes | Comma-separated allowed browser origins (the site URL) |
| `PYTHON_VERSION` | API | Set in blueprint | `3.11.9` |
| `NODE_VERSION` | Static site | Set in blueprint | `24.13.0` |
| `VITE_API_BASE_URL` | Static site | Yes | API URL baked into the bundle at build time |
| `SKIN_MODEL_PATH` / `SKIN_MAPPING_PATH` / `SKIN_PREPROCESSING_PATH` | API | No | Defaults are repo-relative and work on Render |
| `MAX_UPLOAD_BYTES` / `MAX_IMAGE_PIXELS` / `MAX_EXPLANATION_IMAGE_DIMENSION` | API | No | Upload safety limits (defaults: 10 MB / 20 MP / 1024 px) |

No secrets are stored in the repository; the only secret (the database
password inside `DATABASE_URL`) is generated by Render and injected at
runtime.

## Compatibility notes and risks

- **torch version**: the project was trained/validated on torch 2.14.0
  (Windows, CPU). PyPI publishes **no Linux x86_64 wheel for 2.14.0**
  (Windows and aarch64 only), which would break the Render build, so
  `requirements.txt` pins the patch release **2.14.1** with torchvision
  **0.29.1** (requires `torch>=2.14.0`). This is a same-series patch bump;
  the trained `best_model.pt` artifact is untouched and loads identically.
- **Memory**: importing torch plus running Grad-CAM is tight on 512 MB
  instances. The blueprint uses the `starter` plan; if you see OOM kills or
  health-check timeouts, upgrade the API service to `standard` (2 GB).
- **Free PostgreSQL** expires about 30 days after creation. For a permanent
  deployment, change the database `plan` in `render.yaml` (e.g.
  `basic-256mb`) before applying.
- **CPU-only inference**: the API never requires CUDA; MobileNetV2 +
  Grad-CAM run comfortably on CPU at this input size (224×224).
- **Uploads are memory-only**: images are processed from an in-memory
  buffer and never written to disk; only a sanitized filename is stored.

## Educational notice

The model was trained on the HAM10000 dermatoscopy dataset for educational
and research purposes. Predictions must not be used for medical decisions.
The application states this notice prominently in its UI.
