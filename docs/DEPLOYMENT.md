# Deployment

## 1. Environments

The project splits into two runtime profiles because MediaPipe constrains the
interpreter version and nothing else does.

| Profile | Python | Installs | Can do |
|---|---|---|---|
| **Analysis** (default) | 3.10+ (3.14 verified) | `requirements.txt` | Everything except reading video files |
| **Video ingestion** | 3.9–3.12 | `requirements-video.txt` | Video → keypoints |

This is not a workaround, it is the architecture working as intended: pose
estimation sits behind one function and everything downstream consumes
keypoints. In production the split is useful on its own terms — extraction is
GPU/CPU-heavy and bursty, scoring is cheap and constant, so they scale
differently.

```bat
:: Windows
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

:: Video profile (only if you need to ingest .mp4 etc.)
py -3.12 -m venv .venv312
.venv312\Scripts\pip install -r requirements-video.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Video profile. If the interpreter above is 3.9-3.12 there is nothing to split:
# install requirements-video.txt into .venv and skip this.
python3.12 -m venv .venv312
.venv312/bin/pip install -r requirements-video.txt
```

Either way, `run.bat setup` / `./run.sh setup` does the whole thing, including
the platform checks.

## 2. Local run

```powershell
# Windows (PowerShell; cmd: set PYTHONPATH=src)
$env:PYTHONPATH = "src"

python -m runway_ai info                       # system status
python -m runway_ai analyze walk.csv --user Alice --save-report
python -m runway_ai progress --user Alice

run.bat web                                    # API + React app together
uvicorn app.api:app --reload                   # http://127.0.0.1:8000/docs
```

```bash
# macOS / Linux
export PYTHONPATH="$PWD/src"

python -m runway_ai info
python -m runway_ai analyze walk.csv --user Alice --save-report
python -m runway_ai progress --user Alice

./run.sh web                                   # API + React app together
uvicorn app.api:app --reload                   # http://127.0.0.1:8000/docs
```

## 3. Containerised

```dockerfile
# Dockerfile - video profile, so one image does everything
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-video.txt .
RUN pip install --no-cache-dir -r requirements-video.txt \
      fastapi "uvicorn[standard]" python-multipart

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]        # persist runway.db
  web:
    # Static bundle: `cd web && npm run build` produces web/dist.
    image: nginx:alpine
    volumes: ["./web/dist:/usr/share/nginx/html:ro"]
    ports: ["8080:80"]
    depends_on: [api]
```

Ship `models/` in the image. The twelve estimators total a few MB and baking
them in keeps the container stateless apart from the database volume.

## 4. Cloud

| Component | Option | Notes |
|---|---|---|
| API | Cloud Run / App Runner / Azure Container Apps | Scales to zero; a walk is a short request |
| Web app | Any static host (Vercel, Netlify, S3+CDN) — it is a plain SPA bundle | Set `VITE_API_URL` at build time |
| Database | SQLite on a volume → managed Postgres | `db/store.py` is plain SQL; the port is a connection string plus a dialect pass |
| Object storage | S3 / GCS / Blob | Store **keypoints, not video**, wherever consent allows |
| Models | In-image, or a registry (MLflow) if retraining often | |

### Scaling notes

* Pose extraction dominates cost: roughly real-time per clip on CPU. Put it on a
  queue (Celery / Cloud Tasks) rather than in the request path, and return a job
  id.
* Feature extraction and scoring are milliseconds. `app/api.py` loads the models
  once into a module-level cache; loading twelve estimators per request would
  otherwise dominate latency.
* SQLite is fine for single-instance and coursework demos. It is not fine for
  concurrent writers — move to Postgres before running more than one API
  replica.

## 5. Retraining

```bash
python scripts/build_dataset.py --n 3000 --seed 7
python scripts/calibrate_rubric.py
python scripts/train_model.py
python -m pytest tests -q
```

Retraining rewrites `models/*.joblib`, `models/training_report.json` and
`models/rubric_calibration.json`. Gate any deployment on the test suite passing
*and* on `reports/model_comparison.csv` still showing the model beating the
rubric on every criterion — that comparison is the regression check that
matters, because a model that quietly degrades below its own rule-based baseline
is worse than useless.

Version the models alongside the feature list. Each `.joblib` bundle stores the
exact `features` list it was trained with, and `predict_scores` builds its input
frame from that list, so a feature added or reordered in `builder.py` cannot
silently misalign columns at inference.

## 6. Privacy and data handling

Runway footage is identifiable biometric data. Three practices follow:

* **Store keypoints, not video.** The pipeline never needs the pixels again
  after extraction; keypoints are far less identifiable and much smaller.
* **Consent and deletion.** Record consent per subject and support hard delete;
  the schema's `ON DELETE CASCADE` on `sessions`, `session_scores`,
  `session_features` and `findings` makes removing a user remove everything
  derived from them.
* **No demographic inference.** The system scores technique. It must not be
  extended to infer age, body type, ethnicity or attractiveness, and the feature
  set is deliberately restricted to kinematic quantities in normalised units.
