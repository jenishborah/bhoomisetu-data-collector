# BhoomiSetu Data Collector

### Government Land Acquisition Data Collection, Normalization & Temporal Intelligence Pipeline

BhoomiSetu is an AI-assisted land acquisition intelligence platform being developed for **Smart India Hackathon 2026**.

The objective of BhoomiSetu is to move land acquisition management from a largely reactive process toward a **predictive, evidence-backed and decision-oriented system**.

The broader BhoomiSetu platform is designed to answer:

> **Which land acquisition projects are likely to experience delays, why are they at risk, what dependencies are causing the risk, and what action can be taken early?**

This repository contains the **data collection and data engineering foundation** of BhoomiSetu.

---

# 1. What This Repository Does

This repository is responsible for converting publicly accessible and authorized government land-acquisition information into a structured, machine-readable dataset suitable for downstream analytics, GIS processing and machine-learning development.

The current pipeline focuses on **BhoomiRashi**, the Ministry of Road Transport & Highways land acquisition portal.

The pipeline currently performs:

1. Government project discovery
2. Project metadata extraction
3. Notification extraction
4. Sanction information extraction
5. 3a notification detail extraction
6. 3D survey and land-party extraction
7. Data normalization
8. Feature engineering
9. Temporal timeline construction
10. Project-level dataset construction
11. Validation and data-quality checks

The resulting datasets form the initial data foundation for the future BhoomiSetu predictive analytics system.

---

# 2. BhoomiSetu Architecture

The overall BhoomiSetu system is planned as:

```text
                    GOVERNMENT DATA
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     BhoomiRashi      Land Records       GIS / EO
          │                │                │
          │          State Systems         │
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                 SOURCE CONNECTORS
                           │
                           ▼
                  RAW / PROVENANCE
                       LAYER
                           │
                           ▼
                  NORMALIZATION
                       LAYER
                           │
                           ▼
               CANONICAL DATA MODEL
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      FEATURE ENGINEERING         TEMPORAL EVENTS
             │                           │
             └─────────────┬─────────────┘
                           ▼
                BHOOMISETU DATA LAYER
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
            GIS           ML          ANALYTICS
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  DECISION INTELLIGENCE
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Risk Score     Evidence       Next Best
                       Drivers          Action
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                     BHOOMISETU
                      PLATFORM
```

---

# 3. Current Data Pipeline

The currently implemented pipeline is:

```text
BhoomiRashi
     │
     ▼
Project Collection
     │
     ├── Project Manifest
     ├── Notifications
     ├── Sanctions
     ├── 3a Details
     └── 3D Details
            │
            ▼
      Normalization
            │
            ▼
     Feature Engineering
            │
            ├── Land features
            ├── Survey features
            ├── Stakeholder features
            └── Acquisition progress
            │
            ▼
     Timeline Construction
            │
            ├── 3a → 3A
            ├── 3A → 3D
            ├── 3a → 3D
            └── Notification gaps
            │
            ▼
      Project Master Dataset
```

---

# 4. Current Government Data Source

## BhoomiRashi

The current primary government source is **BhoomiRashi**, the Ministry of Road Transport & Highways land acquisition system.

The portal contains project and land-acquisition information associated with National Highway projects.

BhoomiRashi provides information related to:

- Projects
- Project locations
- Land requirement
- Land availability
- Land acquisition
- Notifications
- 3a notifications
- 3A notifications
- 3D notifications
- Survey numbers
- Land parcels
- Owners / parties
- Objections
- Compensation-related information
- Competent Authority for Land Acquisition (CALA)
- Project sanctions
- File and workflow information

The BhoomiSetu collector uses only information that is publicly accessible or otherwise authorized for collection.

**The collector does not bypass CAPTCHA, authentication, access controls or restricted government interfaces.**

---

# 5. Understanding the 3a → 3A → 3D Workflow

The notification lifecycle is an important temporal signal for BhoomiSetu.

The collector preserves the notification sequence rather than treating a project as a static record.

Conceptually:

```text
Project
   │
   ▼
3a Notification
   │
   │
   ├── location / village context
   ├── preliminary acquisition information
   │
   ▼
3A Notification
   │
   │
   ├── survey information
   ├── objections / related processing
   │
   ▼
3D Declaration
   │
   │
   ├── finalized acquisition information
   ├── survey records
   ├── parties / owners
   │
   ▼
Acquisition / Compensation / Possession
```

The actual project may contain multiple notifications at each stage.

For example, project `54635` contains:

```text
3a : 3 notifications
3A : 3 notifications
3D : 3 notifications
```

Therefore, the system preserves individual notification events rather than collapsing them into a single date.

---

# 6. Normalized Data Model

The collector converts source-specific information into normalized CSV structures.

## Project Manifest

Contains project-level information such as:

```text
project_id
project_name
project_number
state
district
land_required_ha
land_available_ha
land_to_acquire_ha
land_acquired_ha
...
```

---

## Notifications

The notification table uses:

```text
project_id
notification_id
notification_type
serial_number
publish_date
notification_number
status
details_url
objections_url
```

The `notification_type` identifies:

```text
3a
3A
3D
```

---

## 3D Survey Data

The normalized 3D survey structure contains:

```text
project_id
notification_id
notification_number
serial_number
district
sub_district
village
survey_number
survey_number_raw
area_hectares
area_raw
land_type
land_nature
land_category
description_raw
party_count
owner_count
affected_party_count
total_party_area_hectares
```

---

## 3D Land Party Data

The normalized party structure contains:

```text
project_id
notification_id
notification_number
survey_serial_number
district
sub_district
village
survey_number
party_sequence
party_name
party_address
party_type
party_area_hectares
party_area_raw
```

Missing party-area values are preserved as missing values.

They are **not converted to zero**, because missing information and zero land area have different meanings.

---

# 7. Feature Engineering

The feature-engineering layer transforms normalized records into project-level analytical features.

Examples include:

### Land and acquisition

```text
land_required_ha
land_to_acquire_ha
land_acquired_ha
acquisition_completion_pct
```

### Survey coverage

```text
survey_count
surveyed_area_ha
surveyed_vs_land_to_acquire_pct
private_area_ha
government_area_ha
private_land_pct
government_land_pct
```

### Geographic distribution

```text
village_count
district_count
urban_survey_count
urban_survey_pct
```

### Stakeholder / party indicators

```text
party_record_count
owner_record_count
affected_party_count
affected_party_rate_pct
party_area_numeric_count
party_area_missing_count
party_area_total_ha
party_area_mean_ha
party_area_max_ha
```

These features provide the foundation for future risk modelling.

---

# 8. Temporal Intelligence

Land acquisition is not a static process.

A project can look healthy at one point in time and become increasingly delayed later.

Therefore, BhoomiSetu preserves temporal information.

The timeline layer derives:

```text
first_3a_date
latest_3a_date

first_3A_date
latest_3A_date

first_3D_date
latest_3D_date

days_3a_to_3A
days_3A_to_3D
days_3a_to_3D

timeline_span_days
```

It also preserves individual notification events.

The event-level timeline contains:

```text
notification_id
notification_type
publish_date
previous_publish_date
days_since_previous_event
days_since_first_event
```

This allows BhoomiSetu to detect long periods of inactivity between acquisition events.

For example, the current reference data contains projects with substantially different transition durations.

These differences are useful temporal signals for future modelling.

---

# 9. Current Prototype Dataset

The current prototype contains five BhoomiRashi project references:

```text
54635
59362
60432
60681
61053
```

The current extracted 3D data contains:

```text
508 survey records
751 party records
```

The project-level feature table contains:

```text
5 projects
32 engineered features
```

The combined project master currently contains:

```text
5 projects
50 columns
```

These records are being used as a **data-engineering and integration reference set**.

They are **not sufficient by themselves to train a production machine-learning model**.

The test project `59362` should also not be interpreted as a real-world training example.

---

# 10. Why We Do Not Train the ML Model Directly on These Five Projects

BhoomiSetu ultimately needs to answer a temporal question:

> Given what is known about a project at a particular point in time, what is the probability that the project will experience a delay in the coming period?

A single static row per project is therefore insufficient.

The future ML dataset will use:

```text
project_id
snapshot_date
current_stage
days_in_current_stage
historical features
current operational features
historical notification information
GIS/context features
...
```

followed by future-looking labels such as:

```text
delay_next_30d
delay_next_60d
delay_next_90d
```

The labels will be generated only from events that occur **after the snapshot date**.

This prevents future information from leaking into the model inputs.

---

# 11. From Static Project Data to Temporal Snapshots

The future training architecture will transform:

```text
PROJECT
  │
  ├── Event 1
  ├── Event 2
  ├── Event 3
  ├── Event 4
  └── Event 5
```

into:

```text
PROJECT SNAPSHOT 1
PROJECT SNAPSHOT 2
PROJECT SNAPSHOT 3
PROJECT SNAPSHOT 4
...
```

Each snapshot represents what BhoomiSetu knew at that point in time.

Example:

```text
Snapshot Date: 2024-03-06

Current Stage:
3A

Known History:
3a notification
3A notification

Current Indicators:
land acquisition progress
stakeholder indicators
administrative indicators
GIS indicators

Future Outcome:
Did the project experience a delay in the next 30/60/90 days?
```

This temporal structure will later support Stage Sentinel.

---

# 12. Scaling the Data Collection Architecture

The current implementation starts with BhoomiRashi, but the architecture is intentionally designed around **source connectors**.

Instead of making the entire system dependent on one website, each government source will have its own connector.

Conceptually:

```text
                    BhoomiSetu
                        │
                Source Connector Layer
                        │
        ┌───────────────┼────────────────┐
        │               │                │
   BhoomiRashi      State Land        GIS / EO
                    Systems           Services
        │               │                │
        │          ILRMS / Landhub      │
        │          BhuNaksha            │
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
                Canonical Data Model
```

Each connector should be responsible for:

- source authentication where authorized
- source-specific request handling
- source-specific parsing
- source metadata
- provenance
- rate limiting
- retry handling
- validation
- normalization

The downstream BhoomiSetu system should not need to know the internal structure of each government portal.

---

# 13. Planned Government Data Integration

As BhoomiSetu scales, multiple official government information sources can contribute complementary context.

Potential sources include:

### Land Records

State land-record systems can provide:

- cadastral information
- survey / Dag numbers
- land ownership context
- land classification
- parcel information

For Assam, the broader ILRMS ecosystem includes services such as:

```text
Dharitree
BhuNaksha
Landhub
Basundhara
```

Where official APIs or authorized interfaces are available, BhoomiSetu can integrate them through dedicated connectors.

---

### GIS and Earth Observation

Geospatial information can enrich acquisition projects with:

- land-use / land-cover
- water bodies
- flood-prone areas
- terrain
- environmental context
- settlement proximity
- infrastructure context

BhoomiSetu can consume official geospatial services such as OGC-compatible WMS/WMTS services where permitted.

---

### Environmental and Forest Clearances

Environmental workflows can provide context around:

- environmental clearance
- forest-related approvals
- wildlife-related considerations
- CRZ-related considerations

These signals can become part of the project's dependency and regulatory context.

---

### Census / Demographic Information

Official demographic datasets can provide aggregated context such as:

- households
- population
- workforce characteristics
- village-level demographic indicators

These should be used at appropriate geographic aggregation levels and with responsible interpretation.

---

### Open Government Data

The Open Government Data platform can provide machine-readable datasets that complement project and administrative information.

Where an official API or downloadable dataset exists, it should be preferred over website extraction.

---

# 14. API-First Integration Strategy

BhoomiSetu should not rely on web scraping as the long-term architecture.

The integration priority is:

```text
1. Official API
       ↓
2. Official machine-readable dataset
       ↓
3. Official OGC / GIS service
       ↓
4. Authorized government data feed
       ↓
5. Public portal extraction where permitted
```

The source connector should hide the implementation details.

For example:

```text
BhoomiRashiConnector
        │
        ▼
Raw BhoomiRashi records
        │
        ▼
BhoomiSetu Normalizer
        │
        ▼
Canonical project / parcel / event model
```

If an official BhoomiRashi API becomes available in the future, the connector can be replaced or upgraded without changing the ML and analytics layers.

---

# 15. Provenance and Data Lineage

Government data should remain traceable.

Every normalized record should ultimately be associated with its source context wherever available.

The planned lineage is:

```text
Source
  ↓
URL / API / Dataset
  ↓
Collection timestamp
  ↓
Raw record
  ↓
Parser
  ↓
Normalized record
  ↓
Feature
  ↓
ML / Analytics output
```

This is important because BhoomiSetu's predictions must be explainable and auditable.

A future risk explanation should be able to answer:

> "Why was this project classified as high risk?"

and:

> "Which underlying project information contributed to this assessment?"

---

# 16. Responsible Government Data Collection

BhoomiSetu follows a responsible integration approach.

The collector should:

- use publicly accessible or authorized information
- respect authentication boundaries
- respect CAPTCHA mechanisms
- avoid bypassing access controls
- avoid unauthorized API access
- avoid aggressive request rates
- preserve source provenance
- validate extracted records
- distinguish missing information from zero values

The system should never attempt to circumvent restrictions imposed by a government system.

---

# 17. Data Quality

Data quality is treated as a first-class part of the pipeline.

Validation includes:

- schema validation
- required-field checks
- date validation
- duplicate detection
- project ID consistency
- notification ID consistency
- numeric parsing
- missing-value analysis
- cross-table consistency
- survey/party relationship checks

Example:

```text
Project
   │
   ├── Notification
   │       │
   │       └── 3D
   │             │
   │             ├── Survey
   │             │
   │             └── Party
   │
   └── Sanction
```

Relationships between these entities must remain consistent.

---

# 18. Repository Structure

```text
bhoomisetu-data-collector/
│
├── connectors/
│   ├── __init__.py
│   └── test_bhoomirashi.py
│
├── extractors/
│   ├── bhoomirashi_project.py
│   ├── bhoomirashi_3a.py
│   ├── bhoomirashi_3D.py
│   ├── bhoomirashi_html.py
│   ├── project_features.py
│   ├── stage_timeline.py
│   └── validation / inspection utilities
│
├── normalize/
│
├── provenance/
│
├── validate/
│
├── scripts/
│   ├── build_project_features.py
│   ├── build_project_timeline.py
│   └── build_project_master.py
│
├── tests/
│
├── output/
│   └── normalized/
│       └── bhoomirashi/
│
├── ingest.py
├── requirements.txt
├── validate_bhoomirashi.py
├── .gitignore
└── README.md
```

---

# 19. Reproducing the Current Pipeline

## 1. Clone the repository

```bash
git clone <repository-url>
cd bhoomisetu-data-collector
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Run the collection / extraction pipeline

The project-specific extractors are located under:

```text
extractors/
```

and the reusable dataset-building scripts are under:

```text
scripts/
```

## 5. Build project features

```powershell
python scripts\build_project_features.py
```

Output:

```text
output\normalized\bhoomirashi\project_features.csv
```

## 6. Build the project timeline

```powershell
python scripts\build_project_timeline.py
```

Outputs:

```text
output\normalized\bhoomirashi\project_timeline.csv
output\normalized\bhoomirashi\notification_timeline.csv
```

## 7. Build the project master dataset

```powershell
python scripts\build_project_master.py
```

Output:

```text
output\normalized\bhoomirashi\project_master.csv
```

---

# 20. Current Data Products

The most important generated datasets are:

```text
project_features.csv
```

Project-level engineered features.

```text
project_timeline.csv
```

Project-level temporal features.

```text
notification_timeline.csv
```

Individual notification events and temporal gaps.

```text
project_master.csv
```

Combined project-level analytical dataset.

---

# 21. Evolution Toward the Full BhoomiSetu Platform

This repository represents the **data engineering layer** of the larger BhoomiSetu system.

The planned evolution is:

```text
PHASE 1
Government Data Collection
        │
        ▼
Normalization
        │
        ▼
Feature Engineering
        │
        ▼
Temporal Dataset
```

↓

```text
PHASE 2
Synthetic Temporal Dataset
        │
        ▼
ML Training
        │
        ├── 30-day delay probability
        ├── 60-day delay probability
        └── 90-day delay probability
```

↓

```text
PHASE 3
Stage Sentinel
        │
        ▼
Risk Detection
        │
        ▼
SHAP / Evidence
```

↓

```text
PHASE 4
BhoomiLens
        │
        ├── GIS
        ├── environmental context
        ├── social / livelihood context
        └── regulatory context
```

↓

```text
PHASE 5
Dependency Graph
        │
        ▼
Delay Cascade
        │
        ▼
Intervention Engine
```

↓

```text
PHASE 6
Decision Intelligence
        │
        ▼
Next Best Action
        │
        ▼
Closed-loop Outcome Learning
```

---

# 22. Planned Machine Learning Layer

The data collector is designed to support future predictive models such as:

### Delay Classification

Calibrated gradient-boosting models such as:

```text
XGBoost
LightGBM
```

for short-horizon delay probabilities.

### Time-to-Event Modelling

Survival / time-to-event methods will be used to estimate:

- stage completion risk
- expected time to completion
- ongoing/censored projects

### Explainability

SHAP-based explanations will connect model predictions to measurable project features.

The ML layer will distinguish:

```text
Prediction
```

from:

```text
Evidence
```

and will avoid presenting statistical associations as guaranteed causal relationships.

---

# 23. Synthetic Data Strategy

Because a small number of publicly accessible project records is not sufficient to train a reliable predictive model, the development pipeline will use synthetic temporal data during the prototype stage.

Synthetic data will simulate:

- multiple projects
- multiple temporal snapshots
- project stages
- administrative bottlenecks
- compensation delays
- legal indicators
- documentation completeness
- rehabilitation and resettlement progress
- stakeholder indicators
- geographic/context variables
- future delay outcomes

The synthetic generator will preserve realistic relationships between variables while preventing future information from leaking into model inputs.

Real government records will remain valuable for:

- schema validation
- feature calibration
- pipeline testing
- geographic validation
- real-world demonstration
- later model validation when sufficient historical data becomes available

---

# 24. Scaling From Prototype to National Deployment

The long-term system is intended to support:

```text
District
   ↓
State
   ↓
Multiple States
   ↓
National Land Acquisition Intelligence
```

Scaling will happen through modular connectors.

Instead of creating a separate application for every government system:

```text
                    BhoomiSetu
                        │
                 Canonical Model
                        │
          ┌─────────────┼─────────────┐
          │             │             │
     BhoomiRashi    Assam ILRMS    Other State
          │             │             │
       Connector      Connector      Connector
```

All connectors feed the same canonical model.

This allows the analytics and ML layers to remain source-independent.

---

# 25. Production Data Architecture

At production scale, the CSV-based prototype can evolve toward:

```text
Government Sources
       │
       ▼
Ingestion Services
       │
       ▼
Raw Object Storage
       │
       ▼
Validation
       │
       ▼
Canonical Database
       │
       ├── PostgreSQL
       └── PostGIS
       │
       ▼
Feature Store / Analytical Layer
       │
       ▼
ML Services
       │
       ▼
FastAPI / REST APIs
       │
       ▼
BhoomiSetu Web Platform
```

PostGIS will provide the foundation for spatial project, parcel and environmental analysis.

---

# 26. Security and Governance

A production BhoomiSetu deployment should include:

- role-based access control
- API authentication
- audit logging
- encrypted data transport
- secure secrets management
- source-level permissions
- data lineage
- model versioning
- prediction versioning
- human review workflows

Government data should not simply be exposed as an unrestricted public dataset.

Access should depend on the sensitivity and authorization requirements of the source.

---

# 27. Current Status

### Completed

- BhoomiRashi project collection
- Project normalization
- Notification normalization
- Sanction normalization
- 3a extraction
- 3D survey extraction
- 3D party extraction
- Data validation
- Project feature engineering
- Project timeline generation
- Notification timeline generation
- Project master dataset generation

### In Progress / Next

- Temporal synthetic training dataset
- ML feature/label design
- Stage Sentinel
- 30/60/90-day delay prediction
- Model calibration
- SHAP explanations
- GIS enrichment
- BhoomiLens
- Dependency graph
- Delay cascade simulation
- Intervention engine
- Decision dashboard

---

# 28. Disclaimer

This repository is part of a prototype / hackathon development effort for **Smart India Hackathon 2026**.

The currently collected project records are intended for:

- prototype development
- data-pipeline validation
- feature engineering
- demonstration
- research and experimentation

They should not be interpreted as a complete national land-acquisition dataset.

Predictive models developed using synthetic data are prototype models and should not be interpreted as production-validated government decision systems without sufficient real historical data, validation, governance and domain review.

---

# 29. Vision

BhoomiSetu aims to transform land acquisition from:

```text
Reactive
    ↓
Delay occurs
    ↓
Problem is discovered
    ↓
Action is taken
```

into:

```text
Continuous Data
      ↓
Project Understanding
      ↓
Early Risk Detection
      ↓
Evidence-backed Explanation
      ↓
What-if Analysis
      ↓
Recommended Intervention
      ↓
Outcome Tracking
      ↓
Continuous Learning
```

### From Land Data to Construction-Ready Decision

**BhoomiSetu**