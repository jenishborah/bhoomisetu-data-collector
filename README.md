# BhoomiSetu

## Land Acquisition Intelligence & Decision Support Platform

BhoomiSetu is a Smart India Hackathon 2026 prototype for **predictive
land-acquisition monitoring**.

The platform is designed to move land-acquisition management from a
reactive workflow to a **predictive, evidence-backed and
decision-oriented system**.

> **From Land Data to Construction-Ready Decision**

BhoomiSetu helps answer:

-   Which projects are at risk of delay?
-   What factors are driving the risk?
-   Which acquisition stage or dependency needs attention?
-   What evidence supports the risk signal?
-   What action should be considered next?
-   How might an intervention change the projected risk?

This repository contains the data pipeline, temporal project data, ML
prototype, FastAPI backend and React/Vite dashboard used for the
BhoomiSetu demonstration.

------------------------------------------------------------------------

## Live Prototype

### Web Dashboard

**https://bhoomisetu-data-collector.vercel.app**

The deployed frontend connects to the BhoomiSetu API hosted on Render.

### Backend API

**https://bhoomisetu-api-dmcx.onrender.com**

### Interactive API Documentation

**https://bhoomisetu-api-dmcx.onrender.com/docs**

The `/docs` page provides the interactive Swagger/OpenAPI interface for
testing the available endpoints.

### Source Repository

**https://github.com/jenishborah/bhoomisetu-data-collector**

------------------------------------------------------------------------

# 1. What is BhoomiSetu?

Land acquisition involves multiple stages, documents, approvals,
compensation activities, rehabilitation and resettlement processes,
legal dependencies and field-level progress.

A project may appear healthy at one point and become increasingly
exposed to delay later.

BhoomiSetu therefore treats acquisition as a **temporal decision
problem**, rather than a static project-status problem.

The prototype combines:

``` text
Project Data
     ↓
Temporal Project State
     ↓
Feature Engineering
     ↓
ML Risk Prediction
     ↓
Risk Bands
     ↓
Evidence / Explanation
     ↓
Recommended Actions
     ↓
What-if Simulation
     ↓
Decision Support
```

------------------------------------------------------------------------

# 2. Core Capabilities

## 2.1 National Dashboard

Provides a national monitoring view containing:

-   Total projects
-   Ongoing projects
-   Risk distribution
-   State-wise risk distribution
-   Stage-wise project distribution
-   Agency-wise project distribution
-   Recently flagged high-priority projects
-   30/60/90-day risk signals

------------------------------------------------------------------------

## 2.2 Project Monitoring

Each project can be inspected individually for:

-   Current acquisition stage
-   Project metadata
-   Temporal snapshots
-   Risk score
-   Risk band
-   30-day delay probability
-   60-day delay probability
-   90-day delay probability
-   Risk trend

------------------------------------------------------------------------

## 2.3 Predictive Risk Engine

The prototype uses XGBoost models for three prediction horizons:

``` text
30 days
60 days
90 days
```

The raw model probabilities are passed through the existing sigmoid
calibration layer before being used as the prototype risk probabilities.

The headline project risk band is derived from the calibrated 90-day
risk.

------------------------------------------------------------------------

## 2.4 Evidence and Explainability

The prototype includes saved SHAP-based project explanations and global
model context.

The goal is not only to say:

> "This project is high risk."

but to provide evidence for:

> "Why is this project currently being flagged?"

------------------------------------------------------------------------

## 2.5 Recommended Actions

The action engine produces deterministic workflow recommendations based
on the current project state.

Examples include actions related to:

-   pending approvals
-   documentation gaps
-   compensation progress
-   R&R progress
-   grievance handling
-   legal dependencies
-   stage-level bottlenecks

These are **decision-support prompts**, not guaranteed outcomes.

------------------------------------------------------------------------

## 2.6 What-if Simulation

The simulator allows selected project conditions to be changed and the
model to be rerun to explore a possible intervention scenario.

Conceptually:

``` text
Current Project
      ↓
Baseline Risk
      ↓
Change Selected Inputs
      ↓
Re-score Project
      ↓
Compare Risk
```

------------------------------------------------------------------------

# 3. How the System Works

## End-to-end flow

``` text
                 GOVERNMENT / PROJECT DATA
                           │
                           ▼
                  Data Collection Layer
                           │
                           ▼
                  Normalization Layer
                           │
                           ▼
                  Temporal Data Layer
                           │
                           ▼
                   Feature Engineering
                           │
                           ▼
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
                 GIS                ML
                                    │
                                    ▼
                         30 / 60 / 90 Day Risk
                                    │
                                    ▼
                         Risk Classification
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 Evidence        Actions        Simulation
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                           Decision Dashboard
```

------------------------------------------------------------------------

# 4. ML Pipeline

The current prototype uses:

-   XGBoost
-   scikit-learn preprocessing
-   Sigmoid probability calibration
-   SHAP-based explanation artifacts
-   Temporal/project-level validation

The prototype prediction horizons are:

  Horizon   Target
  --------- ------------------
  30 days   `delay_next_30d`
  60 days   `delay_next_60d`
  90 days   `delay_next_90d`

### Important

The current ML data is **synthetic prototype data** created for workflow
and demonstration validation.

The model should **not** be interpreted as a production-grade government
forecasting model.

The repository intentionally exposes the prototype limitation through
the API and UI.

------------------------------------------------------------------------

# 5. Risk Bands

The prototype uses four headline risk bands:

  -----------------------------------------------------------------------
  Risk                                Interpretation
  ----------------------------------- -----------------------------------
  LOW                                 Lower current modeled delay
                                      exposure

  MODERATE                            Moderate modeled delay exposure

  HIGH                                Elevated modeled delay exposure
                                      requiring attention

  CRITICAL                            Very high modeled delay exposure
                                      requiring priority attention
  -----------------------------------------------------------------------

The exact numerical probability is also returned by the API.

Risk estimates are calibrated model probabilities, not guarantees that a
project will or will not be delayed.

------------------------------------------------------------------------

# 6. Repository Structure

``` text
bhoomisetu-data-collector/
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       │
│       ├── repositories/
│       │   └── project_repository.py
│       │
│       ├── routers/
│       │   ├── projects.py
│       │   └── dashboard.py
│       │
│       └── services/
│           ├── predictor.py
│           ├── risk_engine.py
│           ├── evidence_engine.py
│           ├── action_engine.py
│           ├── dashboard_service.py
│           └── simulator.py
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.*
│
├── output/
│   └── synthetic/
│       └── ml/
│           ├── baseline/
│           │   └── models/
│           └── calibration/
│               └── models/
│
├── scripts/
├── validate/
├── tests/
├── requirements.txt
├── ingest.py
└── README.md
```

------------------------------------------------------------------------

# 7. ML Model Artifacts

The backend expects the trained prototype artifacts at:

``` text
output/synthetic/ml/baseline/models/
```

with:

``` text
delay_next_30d_xgboost.joblib
delay_next_60d_xgboost.joblib
delay_next_90d_xgboost.joblib
```

and the calibration artifacts at:

``` text
output/synthetic/ml/calibration/models/
```

with:

``` text
delay_next_30d_sigmoid.joblib
delay_next_60d_sigmoid.joblib
delay_next_90d_sigmoid.joblib
```

The predictor service loads these artifacts without retraining the
models.

------------------------------------------------------------------------

# 8. Local Setup

## Prerequisites

Recommended environment:

-   Python 3.11+ / compatible Python environment
-   Node.js 18+
-   npm
-   Git

The backend dependencies are defined in:

``` text
requirements.txt
```

------------------------------------------------------------------------

## Clone the repository

``` bash
git clone https://github.com/jenishborah/bhoomisetu-data-collector.git
cd bhoomisetu-data-collector
```

------------------------------------------------------------------------

# 9. Run the Backend

Create a virtual environment.

### Windows PowerShell

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start FastAPI:

``` bash
uvicorn backend.app.main:app --reload --port 8000
```

Backend:

``` text
http://127.0.0.1:8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

If port `8000` is already in use:

``` bash
uvicorn backend.app.main:app --reload --port 8001
```

Then use:

``` text
http://127.0.0.1:8001/docs
```

------------------------------------------------------------------------

# 10. Run the Frontend

Open another terminal:

``` bash
cd frontend
npm install
npm run dev
```

Vite normally starts the frontend at:

``` text
http://localhost:5173
```

The deployed frontend is configured to communicate with:

``` text
https://bhoomisetu-api-dmcx.onrender.com/api
```

For local development, the frontend API configuration can be adjusted to
point to:

``` text
http://127.0.0.1:8000/api
```

or, if using the alternate local port:

``` text
http://127.0.0.1:8001/api
```

------------------------------------------------------------------------

# 11. Build the Frontend

For a production build:

``` bash
cd frontend
npm run build
```

The generated production files are placed in:

``` text
frontend/dist/
```

------------------------------------------------------------------------

# 12. API Reference

Base URL:

``` text
https://bhoomisetu-api-dmcx.onrender.com
```

API prefix:

``` text
/api
```

Interactive documentation:

``` text
https://bhoomisetu-api-dmcx.onrender.com/docs
```

------------------------------------------------------------------------

## System

### Health check

``` http
GET /health
```

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/health
```

Expected structure:

``` json
{
  "status": "healthy",
  "service": "bhoomisetu-api"
}
```

------------------------------------------------------------------------

## Dashboard

### National overview

``` http
GET /api/dashboard/overview
```

Provides:

-   total projects
-   ongoing projects
-   national risk summary
-   state-wise project distribution
-   stage distribution
-   agency distribution
-   recently flagged projects
-   last update information

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/dashboard/overview
```

------------------------------------------------------------------------

### National risk overview

``` http
GET /api/dashboard/risk-overview
```

Provides:

-   projects scored
-   risk distribution
-   state-level risk distribution
-   project-level risk records
-   high-priority projects
-   model explanation context
-   model version
-   risk basis

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/dashboard/risk-overview
```

------------------------------------------------------------------------

# 13. Project APIs

## List projects

``` http
GET /api/projects
```

Optional query parameters:

``` text
limit
offset
search
state
stage
```

Example:

``` bash
curl "https://bhoomisetu-api-dmcx.onrender.com/api/projects?limit=20"
```

Search example:

``` bash
curl "https://bhoomisetu-api-dmcx.onrender.com/api/projects?state=Assam"
```

------------------------------------------------------------------------

## Get a project

``` http
GET /api/projects/{project_id}
```

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635
```

------------------------------------------------------------------------

## Get project snapshots

``` http
GET /api/projects/{project_id}/snapshots
```

Returns the temporal snapshot history available for the project.

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635/snapshots
```

------------------------------------------------------------------------

## Predict project delay risk

``` http
POST /api/projects/{project_id}/predict
```

Returns calibrated:

-   30-day probability
-   60-day probability
-   90-day probability
-   model version
-   current project stage
-   snapshot identifier

Example:

``` bash
curl -X POST \
  https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635/predict
```

------------------------------------------------------------------------

## Get project risk

``` http
GET /api/projects/{project_id}/risk
```

Returns:

-   risk band
-   headline risk percentage
-   30/60/90-day risk
-   trend
-   current stage
-   snapshot identifier

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635/risk
```

------------------------------------------------------------------------

## Get project evidence

``` http
GET /api/projects/{project_id}/evidence
```

Returns the saved local SHAP explanation for the project when the
explanation artifact is available.

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635/evidence
```

------------------------------------------------------------------------

## Get recommended actions

``` http
GET /api/projects/{project_id}/actions
```

Returns deterministic prototype workflow recommendations based on the
current project state.

Example:

``` bash
curl https://bhoomisetu-api-dmcx.onrender.com/api/projects/54635/actions
```

------------------------------------------------------------------------

## Run what-if simulation

``` http
POST /api/projects/{project_id}/simulate
```

The endpoint accepts editable scenario fields defined by the backend
`SimulationRequest` schema and returns the simulated model result.

Use Swagger for the exact request schema:

``` text
https://bhoomisetu-api-dmcx.onrender.com/docs
```

Navigate to:

``` text
Projects
→ POST /api/projects/{project_id}/simulate
```

------------------------------------------------------------------------

# 14. Example API Workflow

A typical project investigation can follow:

``` text
1. List projects
       ↓
2. Select project
       ↓
3. GET /projects/{id}
       ↓
4. GET /projects/{id}/snapshots
       ↓
5. GET /projects/{id}/risk
       ↓
6. POST /projects/{id}/predict
       ↓
7. GET /projects/{id}/evidence
       ↓
8. GET /projects/{id}/actions
       ↓
9. POST /projects/{id}/simulate
```

This mirrors the BhoomiSetu decision-support workflow:

``` text
Locate
  ↓
Understand
  ↓
Predict
  ↓
Explain
  ↓
Simulate
  ↓
Act
```

------------------------------------------------------------------------

# 15. Dashboard Performance

The national dashboard uses **batch ML inference**.

Instead of performing three model predictions independently for every
project:

``` text
1000 projects
×
3 horizons
×
individual inference
```

the dashboard performs batch inference:

``` text
1000 projects
      ↓
30-day model → one batch
60-day model → one batch
90-day model → one batch
      ↓
calibration
      ↓
risk aggregation
```

This is important for deployment on a low-resource environment such as
the Render free instance.

The resulting dashboard risk data is cached in the running backend
process so subsequent dashboard requests do not repeat the full national
scoring operation.

------------------------------------------------------------------------

# 16. Data and Prototype Scope

The repository contains both the data-engineering foundation and the
synthetic ML prototype.

The data-engineering pipeline is designed around government
land-acquisition information and temporal project records.

The current prototype ML system uses synthetic temporal data for
demonstration.

Therefore:

-   prototype probabilities are not production forecasts
-   synthetic data should not be treated as official government data
-   the prototype is not connected to live government decision systems
-   model outputs are decision-support signals, not automatic decisions
-   the system should not be used to make consequential land-acquisition
    decisions without proper validation, governance and authorized data

------------------------------------------------------------------------

# 17. Responsible AI / Governance

BhoomiSetu is intended to support administrators and project teams
rather than replace their judgment.

The system should:

-   provide evidence with risk signals
-   preserve uncertainty
-   make model limitations visible
-   distinguish prediction from recommendation
-   maintain human review for consequential decisions
-   avoid labeling communities as inherently risky
-   use measurable acquisition and administrative indicators instead of
    demographic stereotypes

The prototype is therefore designed as a **decision-support system**,
not an automated decision-maker.

------------------------------------------------------------------------

# 18. Current Prototype Technology Stack

## Frontend

-   React
-   Vite
-   JavaScript
-   Responsive dashboard UI

## Backend

-   Python
-   FastAPI
-   Uvicorn
-   Pydantic

## Machine Learning

-   XGBoost
-   scikit-learn
-   NumPy
-   pandas
-   joblib
-   sigmoid calibration
-   SHAP explanation artifacts

## Data

-   CSV-based prototype datasets
-   temporal project snapshots
-   engineered project features

## Deployment

-   Vercel --- frontend
-   Render --- FastAPI backend

------------------------------------------------------------------------

# 19. Government Data Foundation

The data-engineering foundation is designed around the BhoomiRashi
land-acquisition ecosystem.

The repository's data pipeline is intended to support:

``` text
Project discovery
       ↓
Project metadata
       ↓
Notifications
       ↓
Sanction information
       ↓
3a / 3A / 3D information
       ↓
Survey / parcel information
       ↓
Party / stakeholder records
       ↓
Normalization
       ↓
Feature engineering
       ↓
Temporal project intelligence
```

The pipeline is designed to use publicly accessible or authorized
information and does not bypass authentication, CAPTCHA or access
controls.

------------------------------------------------------------------------

# 20. Development Workflow

Recommended development sequence:

``` text
1. Update / validate data
        ↓
2. Run feature engineering
        ↓
3. Validate generated data
        ↓
4. Train / update prototype models
        ↓
5. Validate model outputs
        ↓
6. Update backend services
        ↓
7. Test FastAPI endpoints locally
        ↓
8. Build frontend
        ↓
9. Test frontend + backend together
        ↓
10. Push to GitHub
        ↓
11. Render deploys backend
        ↓
12. Vercel deploys frontend
```

------------------------------------------------------------------------

# 21. Troubleshooting

## Backend will not start

If you see:

``` text
WinError 10013
```

the requested port may already be in use.

Try:

``` bash
uvicorn backend.app.main:app --reload --port 8001
```

Or find the process using port 8000 on Windows:

``` powershell
netstat -ano | findstr :8000
```

------------------------------------------------------------------------

## Dashboard takes a long time

The first national dashboard request performs batch model inference and
may be slower after a cold start.

Render's free instance can also spin down after inactivity.

After the initial request, the dashboard scoring result is cached in the
backend process.

------------------------------------------------------------------------

## Model files are missing

Check:

``` text
output/synthetic/ml/baseline/models/
output/synthetic/ml/calibration/models/
```

The predictor requires all six prototype artifacts.

------------------------------------------------------------------------

## Frontend cannot reach the API

Check:

``` text
https://bhoomisetu-api-dmcx.onrender.com/health
```

Then check:

``` text
https://bhoomisetu-api-dmcx.onrender.com/docs
```

If the backend is healthy but the browser reports a CORS error, verify
the frontend origin is allowed by the FastAPI CORS configuration.

------------------------------------------------------------------------

# 22. Useful Links

  ----------------------------------------------------------------------------------------------
  Resource                            Link
  ----------------------------------- ----------------------------------------------------------
  Live Prototype                      https://bhoomisetu-data-collector.vercel.app

  Backend API                         https://bhoomisetu-api-dmcx.onrender.com

  API Docs                            https://bhoomisetu-api-dmcx.onrender.com/docs

  API Health                          https://bhoomisetu-api-dmcx.onrender.com/health

  GitHub Repository                   https://github.com/jenishborah/bhoomisetu-data-collector
  ----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 23. Project Status

**Current status: Demonstration Prototype**

Implemented prototype capabilities include:

-   National dashboard
-   Project listing
-   Project detail
-   Temporal snapshots
-   30/60/90-day delay-risk prediction
-   Risk bands
-   Risk overview
-   SHAP evidence artifacts
-   Deterministic recommended actions
-   What-if simulation
-   Batch national dashboard inference
-   FastAPI backend
-   React/Vite frontend
-   Vercel deployment
-   Render deployment

The prototype is intended for **Smart India Hackathon 2026 demonstration
and workflow validation**.

------------------------------------------------------------------------

# 24. Future Production Roadmap

A production implementation would require additional work in:

-   authorized live government-data integrations
-   PostgreSQL/PostGIS
-   authenticated role-based access
-   immutable audit logging
-   document/OCR ingestion with human verification
-   GIS and remote-sensing integrations
-   model retraining with sufficiently large real temporal datasets
-   monitoring and model drift detection
-   calibration monitoring
-   security hardening
-   data governance
-   privacy and access controls
-   government-system interoperability
-   field/mobile workflows
-   production observability

The current repository provides the prototype foundation for these
extensions.

------------------------------------------------------------------------

# 25. License / Usage

This repository is a Smart India Hackathon 2026 prototype.

Before using the system with real government or land-acquisition data,
establish the required authorization, data-use permissions, security
controls and governance processes.

------------------------------------------------------------------------

## BhoomiSetu

**Land Acquisition Intelligence & Decision Support**

> From Land Data to Construction-Ready Decision.

Built as a Smart India Hackathon 2026 prototype.
