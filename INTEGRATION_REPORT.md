# Skin-Lesion AI — Final End-to-End Integration Report

Date: 2026-10-04
Scope: Full integration and testing of the existing trained model, FastAPI backend, static frontend, Grad-CAM explainability, and PostgreSQL prediction-history layer. No retraining, no model replacement, no dataset/model deletion, no fake/demo predictions, no deployment.

## Checklist summary (29 items)

| # | Item | Result |
|---|------|--------|
| 1 | FastAPI backend starts | PASS — uvicorn on 127.0.0.1:8000, single process, clean startup |
| 2 | Frontend starts | PASS — static server on http://localhost:3000 |
| 3 | Frontend ↔ backend communication | PASS — same-origin-free CORS fetches verified in browser |
| 4 | Real skin image upload via frontend | PASS — ISIC_0024310.jpg (mel) uploaded through the form |
| 5 | Image reaches /predict path | PASS — server logged POST /explain 200; multipart parsed server-side |
| 6 | Trained model generates prediction | PASS — MobileNetV2 checkpoint inference, deterministic output |
| 7 | Predicted class + confidence correct | PASS — MEL — Melanoma, 49.66% in UI = exact API value |
| 8 | Probabilities for ALL 7 classes shown | PASS — added "All-class probabilities" list; sums to 100.00%, sorted |
| 9 | Prediction saved to PostgreSQL | BLOCKED — no credentials available; code path verified in unavailable mode |
| 10 | Saved prediction appears in history | BLOCKED live — /predictions degradation (503 + clear message) verified instead |
| 11 | Grad-CAM workflow | PASS — /explain returns Grad-CAM overlay with prediction |
| 12 | Grad-CAM matches actual prediction | PASS — server-side overlay is generated for the predicted class; overlay ≠ original |
| 13 | Invalid image types | PASS — .txt and .gif → 415 with clear message |
| 14 | Corrupted images | PASS — truncated/empty/unreadable → 422 "not a readable image" |
| 15 | Oversized images | PASS — >10,485,760 bytes → 413; >20,000,000 pixels → 413 |
| 16 | Backend/API failure handling | PASS — dead-port test: friendly "Could not reach the API" message, recovery after restart |
| 17 | Database connection failure | PASS — second server with unreachable DB: /health 200, /predictions 503 no-secrets message, /predict still 200 with `x-prediction-storage: unavailable` |
| 18 | Frontend loading states | PASS — button disabled + "Running AI analysis…" status during request |
| 19 | Frontend error states | PASS — error box shown for 4xx/415/422/413 and network failures; history errors separate |
| 20 | Desktop and mobile sizes | PASS — 1100px two-column grid; 375px stacked single column (computed-style verified) |
| 21 | CORS | PASS — allowed origin gets ACAO header; disallowed origin → 400, no ACAO header |
| 22 | Missing environment variables | PASS — safe defaults; DATABASE_URL empty → history "not_configured", API still serves predictions |
| 23 | Hard-coded URLs | PASS — API address is an editable form field (default http://127.0.0.1:8000); no other hardcoded service URLs |
| 24 | Exposed passwords/secrets | PASS — no .env in repo, no credentials in code; .env.example uses YOUR_PASSWORD placeholder; error messages never echo connection strings |
| 25 | No unnecessary dataset files in production app | PASS — app touches only training_output artifacts; dataset/ never imported by API |
| 26 | Model loaded only once | PASS — single lifespan load site; one "Started server process"; repeat requests reuse it |
| 27 | Inference preprocessing matches training | PASS — checkpoint ↔ inference_preprocessing.json parity script proved identical 224 letterbox/BILINEAR/ImageNet mean-std |
| 28 | Class-label mapping correct | PASS — mapping validated against exact 7-class set; UI names match mapping file |
| 29 | No fake prediction data | PASS — all displayed numbers come from live model output; no stubs/mocks in code paths |

## Frontend status
WORKING. Single-file static app (frontend/index.html — HTML/CSS/vanilla JS) served at http://localhost:3000. Upload form with editable API base, loading state, error box, prediction summary, original + Grad-CAM images, all-class probability bars, paginated history table with detail view and clear/refresh. Responsive at desktop and mobile widths. Note: the frontend is deliberately a zero-build static page, not a React+Vite project (see warnings).

## Backend status
WORKING. FastAPI + uvicorn on 127.0.0.1:8000. Endpoints: GET /health, POST /predict, POST /explain, GET /predictions (paginated), GET /predictions/{id}, DELETE /predictions. Validation chain: 415 unsupported type, 422 unreadable, 413 byte/pixel limits. Model loads once in lifespan. Prediction persists only after a successful inference — failed predictions are never stored as successes.

## AI model status
WORKING — untouched as required. training_output/mobilenetv2_ham10000/best_model.pt (torchvision MobileNetV2, 7-class head), best epoch 5, validation accuracy 0.4416, held-out test accuracy 0.4458 (1,505 images). Repeat predictions are deterministic. Preprocessing parity between training and inference verified. Accuracy is moderate; the app labels output educational/research-only.

## Database status
CODE COMPLETE, LIVE SAVE NOT VERIFIED — blocked pending credentials. PostgreSQL layer (psycopg pool, dict rows, autocommit) is fully implemented with three states: ready / not_configured (DATABASE_URL empty) / unavailable (cannot connect). Both degradation states were verified live against a real running server. A live end-to-end save could not be performed: PostgreSQL 16 is running on 127.0.0.1:5433 but requires a password; no pgpass.conf exists and no credentials were provided (guessing credentials was explicitly ruled out). To finish this last check: create `.env` from `.env.example`, set `DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5433/skin_lesion_ai`, create the `skin_lesion_ai` database, run `python src/init_database.py`, restart the API, run `python src/test_db_operations.py`.

## Prediction API status
WORKING. /predict returns predicted_class, predicted_class_name, confidence, full 7-class probabilities, and storage status header. Verified against real dataset images and via the frontend. Error paths return specific, non-leaking messages.

## Grad-CAM status
WORKING. /explain returns the prediction plus a Grad-CAM overlay computed for the predicted class on the same preprocessed tensor (server-side guarantee of correspondence). Overlay image differs from original; both inlined size-capped (MAX_EXPLANATION_IMAGE_DIMENSION=1024). Verified in browser with real images.

## Prediction History status
WORKING in every reachable state. With storage ready it persists successes only and exposes paginated list/detail/delete. With DATABASE_URL empty → "not_configured" (table hidden, explanatory message). With unreachable DB → "unavailable" (/predictions 503, /predict still succeeds, response header `x-prediction-storage: unavailable`). Live ready-state row round-trip blocked pending credentials (see Database status).

## Errors found
1. No UI display of per-class probabilities (checklist item 8) — API returned them; the page never rendered them.
2. Browser network failures surfaced raw "Failed to fetch" with no guidance, in both upload and history paths.
3. `.env.example` described CORS origins as "React frontend origins" although the served frontend is static HTML.
4. An integration test asserted over-broad sanitization behavior that no longer matched the implemented validation (stale expectation).
5. `src/db.py` did not load a local `.env` file, so a correctly written `.env` would have been ignored when launching outside the shell that exported variables.
6. Environment fault: transient C: disk exhaustion (99% full) made large uploads fail with FastAPI's generic 400 "There was an error parsing the body" instead of a size error.

## Errors fixed
1. Added an "All-class probabilities" section to the result view (shared renderer also used by history detail); verified 7 bars, correct sort, sum 100.00%, top bar = headline confidence.
2. Added explicit `TypeError` handling: upload path now shows "Could not reach the API at … Ensure the backend is running…"; history path shows an equivalent message and hides stale content; both recover automatically once the API returns.
3. Corrected `.env.example` wording and documented the local static server origin (http://localhost:3000).
4. Updated the stale test expectation to match the implemented, correct validation behavior.
5. `src/db.py` now loads a git-ignored `.env` if present (credentials still only from environment/.env — never code).
6. Not a code error: freed disk space and documented the failure mode; 9 MB upload → 200, 17 MB → 413 correct limit message once temp space was available.

## Remaining warnings
1. C: drive is 99% full (~3.5 GB free). Uploads over ~1 MB spool to a server-side temp file; with the disk full the client sees a generic 400 instead of a precise 413. Free disk space before heavy use.
2. Live PostgreSQL round-trip (items 9–10) unverified pending credentials — exact one-time setup steps above. Until then the API runs correctly in "not_configured" mode.
3. The frontend is a single-file static HTML/CSS/JS app, not React + Vite as described in the original plan. It fulfills every UI requirement; tell me if you want a true React+Vite port (no deployment involved).
4. Model accuracy is moderate (test accuracy ≈ 44.6% over 7 classes). This is why every surface states the output is an educational/research classification, not a medical diagnosis — the notice is present in the page banner, the API response NOTICE field, and the OpenAPI description.
5. The Python logger's INFO lines from app modules don't appear in the uvicorn log file (only WARNING+ via lastResort); uvicorn's own access/startup lines do. Cosmetic.
6. Two background processes are still running for your convenience: API on port 8000 and frontend on port 3000. Nothing was deployed.

## Complete project structure
```
Skin-Disease-AI/
├── .env.example                  # documented env template (no real secrets)
├── .gitignore
├── requirements.txt
├── api_server.log                # runtime log of the port-8000 server
├── api_endpoint_test_results.json
├── db_operations_test_results.json
├── model_inference_test_report.md
├── training_run.log
├── INTEGRATION_REPORT.md         # this file
├── src/
│   ├── api.py                    # FastAPI app: /health /predict /explain /predictions
│   ├── db.py                     # psycopg pool + history store (env-only credentials)
│   ├── gradcam.py                # Grad-CAM overlay generation
│   ├── predict_image.py          # checkpoint/mapping/preprocessing loading + inference
│   ├── ham10000_data.py          # preprocessing (224 letterbox) + dataset utils
│   ├── prepare_ham10000.py       # leakage-safe 70/15/15 split builder
│   ├── train_mobilenetv2.py      # training script (already run; NOT re-run)
│   ├── init_database.py          # creates predictions table
│   ├── test_api_endpoints.py     # API integration tests
│   └── test_db_operations.py     # DB round-trip tests
├── frontend/
│   └── index.html                # complete static UI (upload, results, Grad-CAM, history)
├── training_output/mobilenetv2_ham10000/
│   ├── best_model.pt             # trained weights (PROTECTED — untouched)
│   ├── class_label_mapping.json  # 7-class mapping (verified)
│   ├── inference_preprocessing.json (verified parity with training)
│   ├── classification_report.json, test_metrics.json, confusion_matrix.csv/.png,
│   ├── training_history.json, training_report.md, accuracy/loss plots
├── dataset/
│   ├── HAM10000/                 # original images (PROTECTED — untouched)
│   ├── prepared/                 # split_manifest.csv, distributions, prep configs
│   └── archive (1).zip, dataset_analysis_report.md
```

## Exact commands required to run the project locally
```bash
# 1. Dependencies (Python 3.11)
pip install -r requirements.txt

# 2. (Optional) PostgreSQL prediction history
#    Copy .env.example to .env and set, e.g.:
#    DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5433/skin_lesion_ai
#    Then create the database once and initialize the table:
python src/init_database.py

# 3. Start the backend (terminal 1)
cd src
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# 4. Start the frontend (terminal 2)
cd frontend
python -m http.server 3000

# 5. Open the app
#    http://localhost:3000  (API address field defaults to http://127.0.0.1:8000)

# 6. Optional test suites
python src/test_api_endpoints.py    # backend must be running
python src/test_db_operations.py    # requires a configured, reachable DATABASE_URL
```

## Educational/research notice — confirmed
The application states clearly in three places that AI output is an educational/research classification result and NOT a medical diagnosis: the page banner ("This is an educational and research AI classification system, not a medical diagnosis or a substitute for clinical assessment."), the API response `notice` field, and the API documentation description. Grad-CAM copy additionally states the overlay shows model influence only and does not prove disease presence.
