from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# BhoomiSetu Synthetic Temporal Data Generator
#
# PURPOSE
# -------
# Generate longitudinal synthetic acquisition-project data
# for:
#   - pipeline development
#   - ML experimentation
#   - XGBoost testing
#   - survival-model testing
#   - dashboard development
#   - explanation testing
#
# IMPORTANT
# ---------
# This synthetic generator does NOT represent:
#   - official Indian delay frequencies
#   - statutory deadlines
#   - actual government project outcomes
#
# Synthetic delay mechanisms exist only to test the
# BhoomiSetu technical pipeline.
# ============================================================


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "synthetic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PROJECTS_FILE = (
    OUTPUT_DIR
    / "synthetic_projects.csv"
)

SNAPSHOTS_FILE = (
    OUTPUT_DIR
    / "synthetic_snapshots.csv"
)

OUTCOMES_FILE = (
    OUTPUT_DIR
    / "synthetic_project_outcomes.csv"
)


# Reproducibility
RANDOM_SEED = 20260914

N_PROJECTS = 1000

MIN_SNAPSHOTS = 8

MAX_SNAPSHOTS = 15

SNAPSHOT_INTERVAL_DAYS = 30


# ============================================================
# Synthetic scenario distribution
#
# These are generation assumptions only.
# They are NOT real-world frequencies.
# ============================================================

SCENARIOS = [
    "NORMAL",
    "ADMINISTRATIVE_BOTTLENECK",
    "LEGAL_DELAY",
    "COMPENSATION_DELAY",
    "DOCUMENTATION_DELAY",
    "RNR_DELAY",
    "MULTI_FACTOR_DELAY",
    "RECOVERY",
]


SCENARIO_PROBABILITIES = [
    0.45,
    0.10,
    0.08,
    0.08,
    0.07,
    0.07,
    0.10,
    0.05,
]


# ============================================================
# Probability that a project experiences a future synthetic
# delay event at some point in the observation horizon.
#
# Again: synthetic testing assumptions only.
# ============================================================

SCENARIO_DELAY_PROBABILITY = {
    "NORMAL": 0.10,
    "ADMINISTRATIVE_BOTTLENECK": 0.65,
    "LEGAL_DELAY": 0.70,
    "COMPENSATION_DELAY": 0.65,
    "DOCUMENTATION_DELAY": 0.60,
    "RNR_DELAY": 0.60,
    "MULTI_FACTOR_DELAY": 0.82,
    "RECOVERY": 0.25,
}


# ============================================================
# Synthetic latent duration thresholds
#
# ONLY used internally to generate synthetic trajectories.
#
# These are NOT legal or empirical BhoomiSetu benchmarks.
# ============================================================

SYNTHETIC_EXPECTED_DURATION = {
    "3a_to_3A": 150,
    "3A_to_3D": 150,
}


# ============================================================
# Static categorical values
# ============================================================

PROJECT_TYPES = [
    "National Highway",
    "Railway",
    "Irrigation",
    "Transmission",
    "Industrial Corridor",
    "Urban Infrastructure",
]


AGENCIES = [
    "NHAI",
    "NHIDCL",
    "State PWD",
    "Railway",
    "Irrigation Department",
    "Power Utility",
]


STATES = [
    "Assam",
    "Bihar",
    "Odisha",
    "West Bengal",
    "Uttar Pradesh",
    "Maharashtra",
    "Jharkhand",
    "Madhya Pradesh",
]


TERRAINS = [
    "Plain",
    "Rolling",
    "Hilly",
    "Coastal",
    "Riverine",
]


URBAN_RURAL = [
    "Urban",
    "Semi-Urban",
    "Rural",
]


APPROVAL_DEPARTMENTS = [
    "Revenue",
    "Forest",
    "Environment",
    "PWD",
    "District Administration",
    "Multiple Departments",
]


# ============================================================
# Random generator
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# Utility functions
# ============================================================

def clamp(
    value,
    low,
    high,
):
    return max(
        low,
        min(high, value),
    )


def sigmoid(value):

    value = clamp(
        value,
        -30,
        30,
    )

    return 1.0 / (
        1.0 + math.exp(-value)
    )


def choose_weighted(
    values,
    probabilities,
):

    return rng.choice(
        values,
        p=probabilities,
    )


def random_date():

    start = pd.Timestamp(
        "2019-01-01"
    )

    end = pd.Timestamp(
        "2025-12-31"
    )

    total_days = int(
        (end - start).days
    )

    offset_days = int(
        rng.integers(
            0,
            total_days + 1,
        )
    )

    return (
        start
        + pd.Timedelta(
            days=offset_days
        )
    )


# ============================================================
# Initial scenario state
# ============================================================

def scenario_initial_state(
    scenario
):

    state = {
        "pending_approvals_count":
            int(
                rng.integers(
                    0,
                    3,
                )
            ),

        "file_pending_days":
            int(
                rng.integers(
                    0,
                    30,
                )
            ),

        "docs_complete_pct":
            float(
                rng.uniform(
                    80,
                    100,
                )
            ),

        "court_cases_count":
            int(
                rng.integers(
                    0,
                    2,
                )
            ),

        "case_age_months":
            0.0,

        "title_disputes_parcels":
            int(
                rng.integers(
                    0,
                    3,
                )
            ),

        "grievances_30d":
            int(
                rng.integers(
                    0,
                    3,
                )
            ),

        "disbursal_pct":
            0.0,

        "rnr_progress_pct":
            0.0,

        "rnr_consent_pct":
            float(
                rng.uniform(
                    60,
                    95,
                )
            ),

        "grievance_redressal_days":
            float(
                rng.uniform(
                    5,
                    25,
                )
            ),
    }

    # --------------------------------------------------------
    # Scenario-specific starting conditions
    # --------------------------------------------------------

    if scenario == "ADMINISTRATIVE_BOTTLENECK":

        state[
            "pending_approvals_count"
        ] = int(
            rng.integers(
                3,
                7,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            rng.integers(
                30,
                90,
            )
        )

        state[
            "docs_complete_pct"
        ] = float(
            rng.uniform(
                65,
                88,
            )
        )

    elif scenario == "LEGAL_DELAY":

        state[
            "court_cases_count"
        ] = int(
            rng.integers(
                2,
                6,
            )
        )

        state[
            "case_age_months"
        ] = float(
            rng.uniform(
                6,
                30,
            )
        )

        state[
            "title_disputes_parcels"
        ] = int(
            rng.integers(
                3,
                12,
            )
        )

        state[
            "grievances_30d"
        ] = int(
            rng.integers(
                2,
                6,
            )
        )

    elif scenario == "COMPENSATION_DELAY":

        state[
            "disbursal_pct"
        ] = float(
            rng.uniform(
                2,
                15,
            )
        )

        state[
            "grievances_30d"
        ] = int(
            rng.integers(
                3,
                8,
            )
        )

        state[
            "grievance_redressal_days"
        ] = float(
            rng.uniform(
                20,
                60,
            )
        )

    elif scenario == "DOCUMENTATION_DELAY":

        state[
            "docs_complete_pct"
        ] = float(
            rng.uniform(
                45,
                75,
            )
        )

        state[
            "pending_approvals_count"
        ] = int(
            rng.integers(
                2,
                5,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            rng.integers(
                20,
                60,
            )
        )

    elif scenario == "RNR_DELAY":

        state[
            "rnr_progress_pct"
        ] = float(
            rng.uniform(
                0,
                8,
            )
        )

        state[
            "rnr_consent_pct"
        ] = float(
            rng.uniform(
                35,
                70,
            )
        )

        state[
            "grievances_30d"
        ] = int(
            rng.integers(
                3,
                8,
            )
        )

        state[
            "grievance_redressal_days"
        ] = float(
            rng.uniform(
                20,
                60,
            )
        )

    elif scenario == "MULTI_FACTOR_DELAY":

        state[
            "pending_approvals_count"
        ] = int(
            rng.integers(
                3,
                7,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            rng.integers(
                40,
                100,
            )
        )

        state[
            "docs_complete_pct"
        ] = float(
            rng.uniform(
                50,
                80,
            )
        )

        state[
            "court_cases_count"
        ] = int(
            rng.integers(
                2,
                5,
            )
        )

        state[
            "case_age_months"
        ] = float(
            rng.uniform(
                6,
                24,
            )
        )

        state[
            "grievances_30d"
        ] = int(
            rng.integers(
                4,
                10,
            )
        )

        state[
            "rnr_progress_pct"
        ] = float(
            rng.uniform(
                0,
                20,
            )
        )

        state[
            "rnr_consent_pct"
        ] = float(
            rng.uniform(
                30,
                65,
            )
        )

    elif scenario == "RECOVERY":

        state[
            "pending_approvals_count"
        ] = int(
            rng.integers(
                3,
                6,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            rng.integers(
                30,
                70,
            )
        )

        state[
            "docs_complete_pct"
        ] = float(
            rng.uniform(
                55,
                80,
            )
        )

        state[
            "grievances_30d"
        ] = int(
            rng.integers(
                3,
                7,
            )
        )

        state[
            "rnr_progress_pct"
        ] = float(
            rng.uniform(
                5,
                25,
            )
        )

    return state


# ============================================================
# Static project generator
# ============================================================

def generate_project(
    project_index
):

    project_id = (
        f"SYN-{project_index:05d}"
    )

    scenario = choose_weighted(
        SCENARIOS,
        SCENARIO_PROBABILITIES,
    )

    project_type = rng.choice(
        PROJECT_TYPES
    )

    agency = rng.choice(
        AGENCIES
    )

    state = rng.choice(
        STATES
    )

    district = (
        f"{state[:3].upper()}-"
        f"{int(rng.integers(1, 40)):02d}"
    )

    terrain = rng.choice(
        TERRAINS
    )

    settlement = rng.choice(
        URBAN_RURAL,
        p=[
            0.25,
            0.25,
            0.50,
        ],
    )

    land_required = float(
        rng.lognormal(
            mean=3.0,
            sigma=0.8,
        )
    )

    land_required = clamp(
        land_required,
        5,
        500,
    )

    total_parcels = int(
        clamp(
            round(
                land_required
                * rng.uniform(
                    5,
                    20,
                )
            ),
            10,
            5000,
        )
    )

    affected_families = int(
        clamp(
            round(
                total_parcels
                * rng.uniform(
                    0.2,
                    0.8,
                )
            ),
            5,
            3000,
        )
    )

    vulnerable_families = int(
        clamp(
            round(
                affected_families
                * rng.uniform(
                    0.05,
                    0.30,
                )
            ),
            0,
            affected_families,
        )
    )

    forest_involved = int(
        rng.random() < 0.12
    )

    proximity_km = float(
        rng.uniform(
            0.5,
            80,
        )
    )

    start_date = random_date()

    land_to_acquire = (
        land_required
        * rng.uniform(
            0.60,
            1.00,
        )
    )

    return {
        "project_id":
            project_id,

        "project_name":
            (
                f"Synthetic "
                f"{project_type} Project "
                f"{project_index:05d}"
            ),

        "project_type":
            project_type,

        "implementing_agency":
            agency,

        "state":
            state,

        "district":
            district,

        # Ground truth only.
        # This is intentionally removed from snapshots.
        "scenario":
            scenario,

        "land_required_ha":
            round(
                land_required,
                4,
            ),

        "land_to_acquire_ha":
            round(
                land_to_acquire,
                4,
            ),

        "total_parcels":
            total_parcels,

        "affected_families":
            affected_families,

        "vulnerable_families":
            vulnerable_families,

        "date_initiation":
            start_date.strftime(
                "%Y-%m-%d"
            ),

        "terrain_type":
            terrain,

        "urban_rural":
            settlement,

        "forest_involved":
            forest_involved,

        "proximity_km_to_urban":
            round(
                proximity_km,
                3,
            ),
    }


# ============================================================
# Stage progression
# ============================================================

def determine_stage(
    scenario,
    snapshot_number,
    total_snapshots,
):

    progress = (
        snapshot_number
        / max(
            total_snapshots - 1,
            1,
        )
    )

    # --------------------------------------------------------
    # Delayed scenarios progress more slowly.
    # --------------------------------------------------------

    multipliers = {
        "NORMAL": 1.00,
        "ADMINISTRATIVE_BOTTLENECK": 0.78,
        "LEGAL_DELAY": 0.72,
        "COMPENSATION_DELAY": 0.80,
        "DOCUMENTATION_DELAY": 0.75,
        "RNR_DELAY": 0.78,
        "MULTI_FACTOR_DELAY": 0.62,
        "RECOVERY": 0.92,
    }

    progress *= multipliers.get(
        scenario,
        1.0,
    )

    if progress < 0.33:

        return "3a"

    if progress < 0.66:

        return "3A"

    return "3D"


# ============================================================
# Stage entry date
# ============================================================

def calculate_stage_entry_date(
    project_start,
    snapshot_date,
    scenario,
    stage,
    snapshot_number,
):

    # --------------------------------------------------------
    # We create a deterministic stage history rather than
    # randomly generating stage_entry_date.
    #
    # This keeps:
    #
    # days_in_current_stage =
    # snapshot_date - stage_entry_date
    #
    # --------------------------------------------------------

    if stage == "3a":

        entry_date = project_start

    elif stage == "3A":

        # Approximate synthetic transition point.
        # This is part of synthetic generation only.

        entry_day = int(
            snapshot_number
            * SNAPSHOT_INTERVAL_DAYS
            * 0.65
        )

        entry_date = (
            project_start
            + pd.Timedelta(
                days=entry_day
            )
        )

    else:

        entry_day = int(
            snapshot_number
            * SNAPSHOT_INTERVAL_DAYS
            * 0.65
        )

        entry_date = (
            project_start
            + pd.Timedelta(
                days=entry_day
            )
        )

    if entry_date > snapshot_date:

        entry_date = snapshot_date

    return entry_date


# ============================================================
# Update time-varying state
# ============================================================

def update_state(
    state,
    scenario,
    stage,
):

    state = state.copy()

    # --------------------------------------------------------
    # Baseline natural evolution
    # --------------------------------------------------------

    state[
        "pending_approvals_count"
    ] += int(
        rng.choice(
            [-1, 0, 0, 0, 1]
        )
    )

    state[
        "pending_approvals_count"
    ] = int(
        clamp(
            state[
                "pending_approvals_count"
            ],
            0,
            10,
        )
    )

    state[
        "file_pending_days"
    ] += int(
        rng.integers(
            -5,
            10,
        )
    )

    state[
        "file_pending_days"
    ] = int(
        clamp(
            state[
                "file_pending_days"
            ],
            0,
            365,
        )
    )

    state[
        "docs_complete_pct"
    ] += float(
        rng.uniform(
            -1.5,
            3,
        )
    )

    state[
        "docs_complete_pct"
    ] = clamp(
        state[
            "docs_complete_pct"
        ],
        25,
        100,
    )

    # --------------------------------------------------------
    # Legal evolution
    # --------------------------------------------------------

    if (
        state[
            "court_cases_count"
        ] > 0
    ):

        state[
            "case_age_months"
        ] += (
            SNAPSHOT_INTERVAL_DAYS
            / 30.44
        )

    state[
        "case_age_months"
    ] = clamp(
        state[
            "case_age_months"
        ],
        0,
        120,
    )

    # --------------------------------------------------------
    # Grievances
    # --------------------------------------------------------

    state[
        "grievances_30d"
    ] = int(
        clamp(
            state[
                "grievances_30d"
            ]
            + rng.integers(
                -1,
                2,
            ),
            0,
            20,
        )
    )

    # --------------------------------------------------------
    # Compensation
    # --------------------------------------------------------

    if stage == "3A":

        state[
            "disbursal_pct"
        ] += float(
            rng.uniform(
                0,
                6,
            )
        )

    elif stage == "3D":

        state[
            "disbursal_pct"
        ] += float(
            rng.uniform(
                3,
                10,
            )
        )

    state[
        "disbursal_pct"
    ] = clamp(
        state[
            "disbursal_pct"
        ],
        0,
        100,
    )

    # --------------------------------------------------------
    # R&R
    #
    # Normal projects receive ordinary R&R progression.
    # RNR_DELAY projects progress much more slowly so that
    # incomplete rehabilitation/resettlement remains observable
    # across the trajectory.
    #
    # RECOVERY receives its additional recovery progression in
    # the scenario-specific section below.
    # --------------------------------------------------------

    if stage in [
        "3A",
        "3D",
    ]:

        if scenario == "RNR_DELAY":

            state[
                "rnr_progress_pct"
            ] += float(
                rng.uniform(
                    0,
                    1,
                )
            )

        else:

            state[
                "rnr_progress_pct"
            ] += float(
                rng.uniform(
                    0,
                    7,
                )
            )

    state[
        "rnr_progress_pct"
    ] = clamp(
        state[
            "rnr_progress_pct"
        ],
        0,
        100,
    )

    # ========================================================
    # Scenario-specific evolution
    # ========================================================

    if scenario == "ADMINISTRATIVE_BOTTLENECK":

        state[
            "pending_approvals_count"
        ] = int(
            clamp(
                state[
                    "pending_approvals_count"
                ]
                + rng.choice(
                    [0, 0, 1, 1, 2]
                ),
                0,
                10,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            clamp(
                state[
                    "file_pending_days"
                ]
                + rng.integers(
                    2,
                    12,
                ),
                0,
                365,
            )
        )

    elif scenario == "LEGAL_DELAY":

        if rng.random() < 0.08:

            state[
                "court_cases_count"
            ] = int(
                clamp(
                    state[
                        "court_cases_count"
                    ]
                    + 1,
                    0,
                    10,
                )
            )

        state[
            "title_disputes_parcels"
        ] = int(
            clamp(
                state[
                    "title_disputes_parcels"
                ]
                + rng.choice(
                    [-1, 0, 0, 1]
                ),
                0,
                100,
            )
        )

    elif scenario == "COMPENSATION_DELAY":

        state[
            "disbursal_pct"
        ] = clamp(
            state[
                "disbursal_pct"
            ]
            + rng.uniform(
                -1,
                1.5,
            ),
            0,
            100,
        )

        state[
            "grievances_30d"
        ] = int(
            clamp(
                state[
                    "grievances_30d"
                ]
                + rng.choice(
                    [0, 0, 1, 1, 2]
                ),
                0,
                20,
            )
        )

    elif scenario == "DOCUMENTATION_DELAY":

        state[
            "docs_complete_pct"
        ] = clamp(
            state[
                "docs_complete_pct"
            ]
            + rng.uniform(
                -3,
                0.5,
            ),
            20,
            100,
        )

    elif scenario == "RNR_DELAY":

        # R&R remains deliberately slow under the RNR_DELAY
        # scenario. Progress may fluctuate slightly, but its
        # expected movement is much smaller than the baseline
        # R&R progression applied above. This preserves a
        # realistic recovery possibility without making RNR_DELAY
        # look healthier than NORMAL in aggregate validation.
        state[
            "rnr_progress_pct"
        ] = clamp(
            state[
                "rnr_progress_pct"
            ]
            + rng.uniform(
                -0.5,
                0.8,
            ),
            0,
            100,
        )

        state[
            "rnr_consent_pct"
        ] = clamp(
            state[
                "rnr_consent_pct"
            ]
            + rng.uniform(
                -2,
                1,
            ),
            0,
            100,
        )

    elif scenario == "MULTI_FACTOR_DELAY":

        state[
            "pending_approvals_count"
        ] = int(
            clamp(
                state[
                    "pending_approvals_count"
                ]
                + rng.choice(
                    [0, 1, 1, 2]
                ),
                0,
                10,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            clamp(
                state[
                    "file_pending_days"
                ]
                + rng.integers(
                    2,
                    10,
                ),
                0,
                365,
            )
        )

        state[
            "docs_complete_pct"
        ] = clamp(
            state[
                "docs_complete_pct"
            ]
            + rng.uniform(
                -3,
                0.5,
            ),
            20,
            100,
        )

        state[
            "grievances_30d"
        ] = int(
            clamp(
                state[
                    "grievances_30d"
                ]
                + rng.choice(
                    [0, 1, 1, 2]
                ),
                0,
                20,
            )
        )

    elif scenario == "RECOVERY":

        state[
            "pending_approvals_count"
        ] = int(
            clamp(
                state[
                    "pending_approvals_count"
                ]
                - rng.choice(
                    [0, 0, 1]
                ),
                0,
                10,
            )
        )

        state[
            "file_pending_days"
        ] = int(
            clamp(
                state[
                    "file_pending_days"
                ]
                - rng.integers(
                    0,
                    7,
                ),
                0,
                365,
            )
        )

        state[
            "docs_complete_pct"
        ] = clamp(
            state[
                "docs_complete_pct"
            ]
            + rng.uniform(
                1,
                4,
            ),
            30,
            100,
        )

        state[
            "grievances_30d"
        ] = int(
            clamp(
                state[
                    "grievances_30d"
                ]
                - rng.choice(
                    [0, 0, 1]
                ),
                0,
                20,
            )
        )

        state[
            "rnr_progress_pct"
        ] = clamp(
            state[
                "rnr_progress_pct"
            ]
            + rng.uniform(
                2,
                6,
            ),
            0,
            100,
        )

    return state


# ============================================================
# Risk score
# ============================================================

def calculate_risk(
    project,
    state,
    stage,
):

    risk = -3.5

    risk += (
        state[
            "pending_approvals_count"
        ]
        * 0.18
    )

    risk += (
        state[
            "file_pending_days"
        ]
        * 0.008
    )

    risk += (
        100
        - state[
            "docs_complete_pct"
        ]
    ) * 0.018

    risk += (
        state[
            "court_cases_count"
        ]
        * 0.22
    )

    risk += (
        state[
            "case_age_months"
        ]
        * 0.018
    )

    risk += (
        state[
            "title_disputes_parcels"
        ]
        * 0.04
    )

    risk += (
        state[
            "grievances_30d"
        ]
        * 0.12
    )

    risk += (
        100
        - state[
            "disbursal_pct"
        ]
    ) * 0.003

    risk += (
        100
        - state[
            "rnr_progress_pct"
        ]
    ) * 0.003

    risk += (
        100
        - state[
            "rnr_consent_pct"
        ]
    ) * 0.002

    risk += (
        state[
            "grievance_redressal_days"
        ]
        * 0.008
    )

    if project[
        "forest_involved"
    ]:

        risk += 0.15

    if project[
        "terrain_type"
    ] == "Hilly":

        risk += 0.10

    if project[
        "urban_rural"
    ] == "Urban":

        risk += 0.05

    if stage == "3A":

        risk += 0.10

    elif stage == "3D":

        risk += 0.05

    return sigmoid(
        risk
    )


# ============================================================
# Generate hidden future delay event
# ============================================================

def generate_future_delay_day(
    project,
    total_observation_days,
):

    scenario = project[
        "scenario"
    ]

    base_probability = (
        SCENARIO_DELAY_PROBABILITY[
            scenario
        ]
    )

    # --------------------------------------------------------
    # Project complexity modifies probability.
    # --------------------------------------------------------

    complexity_factor = 0.0

    if (
        project[
            "land_required_ha"
        ] > 100
    ):

        complexity_factor += 0.05

    if (
        project[
            "affected_families"
        ] > 500
    ):

        complexity_factor += 0.05

    if project[
        "forest_involved"
    ]:

        complexity_factor += 0.05

    probability = clamp(
        base_probability
        + complexity_factor,
        0.02,
        0.90,
    )

    # --------------------------------------------------------
    # Some projects never experience a synthetic delay event.
    # --------------------------------------------------------

    has_delay = (
        rng.random()
        < probability
    )

    if not has_delay:

        return None

    # --------------------------------------------------------
    # Delay event occurs sometime during the observation
    # horizon, but not necessarily near the beginning.
    # --------------------------------------------------------

    minimum_day = 60

    maximum_day = max(
        minimum_day + 1,
        total_observation_days - 15,
    )

    if minimum_day >= maximum_day:

        return minimum_day

    # Beta distribution produces events at varied times.
    timing_fraction = rng.beta(
        2.0,
        2.2,
    )

    delay_day = (
        minimum_day
        + timing_fraction
        * (
            maximum_day
            - minimum_day
        )
    )

    return int(
        round(
            delay_day
        )
    )


# ============================================================
# Generate project trajectories
# ============================================================

projects = []

snapshots = []

outcomes = []


for project_index in range(
    1,
    N_PROJECTS + 1,
):

    project = generate_project(
        project_index
    )

    scenario = project[
        "scenario"
    ]

    total_snapshots = int(
        rng.integers(
            MIN_SNAPSHOTS,
            MAX_SNAPSHOTS + 1,
        )
    )

    total_observation_days = (
        (
            total_snapshots
            - 1
        )
        * SNAPSHOT_INTERVAL_DAYS
    )

    project_start = pd.Timestamp(
        project[
            "date_initiation"
        ]
    )

    state = scenario_initial_state(
        scenario
    )

    # --------------------------------------------------------
    # Hidden synthetic future event
    # --------------------------------------------------------

    future_delay_day = (
        generate_future_delay_day(
            project,
            total_observation_days,
        )
    )

    first_delay_day = (
        future_delay_day
    )

    delay_30_events = 0

    delay_60_events = 0

    delay_90_events = 0

    previous_stage = None

    stage_entry_date = project_start

    # --------------------------------------------------------
    # Generate snapshots
    # --------------------------------------------------------

    for snapshot_number in range(
        total_snapshots
    ):

        snapshot_date = (
            project_start
            + pd.Timedelta(
                days=(
                    snapshot_number
                    * SNAPSHOT_INTERVAL_DAYS
                )
            )
        )

        days_since_start = (
            snapshot_number
            * SNAPSHOT_INTERVAL_DAYS
        )

        stage = determine_stage(
            scenario,
            snapshot_number,
            total_snapshots,
        )

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        if snapshot_number > 0:

            state = update_state(
                state,
                scenario,
                stage,
            )

        # ----------------------------------------------------
        # Stage transition
        # ----------------------------------------------------

        if (
            previous_stage is None
            or stage != previous_stage
        ):

            stage_entry_date = (
                calculate_stage_entry_date(
                    project_start,
                    snapshot_date,
                    scenario,
                    stage,
                    snapshot_number,
                )
            )

        if stage_entry_date > snapshot_date:

            stage_entry_date = snapshot_date

        days_in_current_stage = (
            snapshot_date
            - stage_entry_date
        ).days

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_probability = (
            calculate_risk(
                project,
                state,
                stage,
            )
        )

        # ----------------------------------------------------
        # Forward-looking labels
        #
        # One hidden event generates all three horizons.
        #
        # If the future event is:
        #
        #   <= 30 days from snapshot → 30/60/90 = 1
        #   <= 60 days               → 30=0, 60/90=1
        #   <= 90 days               → 30/60=0, 90=1
        #   > 90 days / none         → all 0
        #
        # ----------------------------------------------------

        if future_delay_day is None:

            delay_next_30d = 0
            delay_next_60d = 0
            delay_next_90d = 0

        elif (
            future_delay_day
            <= days_since_start
        ):

            # The event has already occurred.
            delay_next_30d = 0
            delay_next_60d = 0
            delay_next_90d = 0

        else:

            days_until_delay = (
                future_delay_day
                - days_since_start
            )

            delay_next_30d = int(
                days_until_delay <= 30
            )

            delay_next_60d = int(
                days_until_delay <= 60
            )

            delay_next_90d = int(
                days_until_delay <= 90
            )

        delay_30_events += (
            delay_next_30d
        )

        delay_60_events += (
            delay_next_60d
        )

        delay_90_events += (
            delay_next_90d
        )

        # ----------------------------------------------------
        # Censoring
        #
        # Only final snapshots can be censored.
        # ----------------------------------------------------

        is_final_snapshot = (
            snapshot_number
            == total_snapshots - 1
        )

        if is_final_snapshot:

            if scenario in [
                "NORMAL",
                "RECOVERY",
            ]:

                censor_probability = 0.20

            else:

                censor_probability = 0.30

            is_censored = int(
                rng.random()
                < censor_probability
            )

        else:

            is_censored = 0

        # ----------------------------------------------------
        # Compensation
        # ----------------------------------------------------

        compensation_awarded_cr = round(
            project[
                "land_required_ha"
            ]
            * rng.uniform(
                0.05,
                0.40,
            ),
            4,
        )

        compensation_disbursed_cr = round(
            compensation_awarded_cr
            * (
                state[
                    "disbursal_pct"
                ]
                / 100
            ),
            4,
        )

        # ----------------------------------------------------
        # Snapshot
        #
        # NO:
        #   scenario
        #   future_delay_day
        #   latent variables
        #
        # YES:
        #   information available at snapshot time
        # ----------------------------------------------------

        snapshot_id = (
            f"{project['project_id']}_"
            f"S{snapshot_number + 1:02d}"
        )

        snapshot = {
            "snapshot_id":
                snapshot_id,

            "project_id":
                project[
                    "project_id"
                ],

            "snapshot_date":
                snapshot_date.strftime(
                    "%Y-%m-%d"
                ),

            "project_start_date":
                project[
                    "date_initiation"
                ],

            "current_stage":
                stage,

            "stage_entry_date":
                stage_entry_date.strftime(
                    "%Y-%m-%d"
                ),

            "days_in_current_stage":
                int(
                    days_in_current_stage
                ),

            "days_since_project_start":
                int(
                    days_since_start
                ),

            "event_count_so_far":
                int(
                    snapshot_number + 1
                ),

            "land_required_ha":
                project[
                    "land_required_ha"
                ],

            "land_to_acquire_ha":
                project[
                    "land_to_acquire_ha"
                ],

            "total_parcels":
                project[
                    "total_parcels"
                ],

            "affected_families":
                project[
                    "affected_families"
                ],

            "vulnerable_families":
                project[
                    "vulnerable_families"
                ],

            "pending_approvals_count":
                state[
                    "pending_approvals_count"
                ],

            "approval_dept":
                rng.choice(
                    APPROVAL_DEPARTMENTS
                ),

            "file_pending_days":
                state[
                    "file_pending_days"
                ],

            "docs_complete_pct":
                round(
                    state[
                        "docs_complete_pct"
                    ],
                    3,
                ),

            "court_cases_count":
                state[
                    "court_cases_count"
                ],

            "case_age_months":
                round(
                    state[
                        "case_age_months"
                    ],
                    3,
                ),

            "title_disputes_parcels":
                state[
                    "title_disputes_parcels"
                ],

            "grievances_30d":
                state[
                    "grievances_30d"
                ],

            "compensation_awarded_cr":
                compensation_awarded_cr,

            "compensation_disbursed_cr":
                compensation_disbursed_cr,

            "disbursal_pct":
                round(
                    state[
                        "disbursal_pct"
                    ],
                    3,
                ),

            "rnr_progress_pct":
                round(
                    state[
                        "rnr_progress_pct"
                    ],
                    3,
                ),

            "rnr_consent_pct":
                round(
                    state[
                        "rnr_consent_pct"
                    ],
                    3,
                ),

            "grievance_redressal_days":
                round(
                    state[
                        "grievance_redressal_days"
                    ],
                    3,
                ),

            "terrain_type":
                project[
                    "terrain_type"
                ],

            "urban_rural":
                project[
                    "urban_rural"
                ],

            "forest_involved":
                project[
                    "forest_involved"
                ],

            "proximity_km_to_urban":
                project[
                    "proximity_km_to_urban"
                ],

            # ------------------------------------------------
            # Synthetic forward labels
            # ------------------------------------------------

            "delay_next_30d":
                delay_next_30d,

            "delay_next_60d":
                delay_next_60d,

            "delay_next_90d":
                delay_next_90d,

            "is_censored":
                is_censored,
        }

        snapshots.append(
            snapshot
        )

        previous_stage = stage

    # ========================================================
    # Final project outcome
    # ========================================================

    final_snapshot_date = (
        project_start
        + pd.Timedelta(
            days=total_observation_days
        )
    )

    delayed = int(
        future_delay_day is not None
        and future_delay_day
        <= total_observation_days
    )

    if delayed:

        delay_days = int(
            max(
                0,
                total_observation_days
                - future_delay_day,
            )
        )

    else:

        delay_days = 0

    # --------------------------------------------------------
    # Final-stage completion
    # --------------------------------------------------------

    final_stage = stage

    final_is_censored = int(
        snapshots[-1][
            "is_censored"
        ]
    )

    outcomes.append(
        {
            "project_id":
                project[
                    "project_id"
                ],

            "scenario_ground_truth":
                scenario,

            "total_snapshots":
                total_snapshots,

            "observation_start_date":
                project[
                    "date_initiation"
                ],

            "observation_end_date":
                final_snapshot_date.strftime(
                    "%Y-%m-%d"
                ),

            "final_stage":
                final_stage,

            "delayed":
                delayed,

            "delay_days":
                delay_days,

            "first_delay_day":
                first_delay_day,

            "delay_30_event_count":
                delay_30_events,

            "delay_60_event_count":
                delay_60_events,

            "delay_90_event_count":
                delay_90_events,

            "is_censored":
                final_is_censored,

            "synthetic_ground_truth_note":
                (
                    "Synthetic outcome generated "
                    "for model development and "
                    "pipeline testing only; "
                    "not a legal, statutory, or "
                    "empirical real-world benchmark."
                ),
        }
    )

    # --------------------------------------------------------
    # Static project
    # --------------------------------------------------------

    projects.append(
        project
    )


# ============================================================
# DataFrames
# ============================================================

projects_df = pd.DataFrame(
    projects
)

snapshots_df = pd.DataFrame(
    snapshots
)

outcomes_df = pd.DataFrame(
    outcomes
)


# ============================================================
# Sort
# ============================================================

projects_df = (
    projects_df
    .sort_values(
        "project_id"
    )
    .reset_index(
        drop=True
    )
)

snapshots_df = (
    snapshots_df
    .sort_values(
        [
            "project_id",
            "snapshot_date",
        ]
    )
    .reset_index(
        drop=True
    )
)

outcomes_df = (
    outcomes_df
    .sort_values(
        "project_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# Save
# ============================================================

projects_df.to_csv(
    PROJECTS_FILE,
    index=False,
)

snapshots_df.to_csv(
    SNAPSHOTS_FILE,
    index=False,
)

outcomes_df.to_csv(
    OUTCOMES_FILE,
    index=False,
)


# ============================================================
# Validation
# ============================================================

print("=" * 70)
print(
    "BhoomiSetu Synthetic Temporal Data Generator"
)
print("=" * 70)

print()

print(
    f"Random seed        : "
    f"{RANDOM_SEED}"
)

print(
    f"Projects generated : "
    f"{len(projects_df)}"
)

print(
    f"Snapshots generated: "
    f"{len(snapshots_df)}"
)

print(
    f"Outcomes generated : "
    f"{len(outcomes_df)}"
)


# ============================================================
# Snapshot distribution
# ============================================================

print()
print(
    "Snapshots per project:"
)

snapshot_counts = (
    snapshots_df
    .groupby(
        "project_id"
    )
    .size()
)

print(
    snapshot_counts.describe()
    .to_string()
)


# ============================================================
# Scenario distribution
# ============================================================

print()
print(
    "Scenario distribution:"
)

scenario_counts = (
    projects_df[
        "scenario"
    ]
    .value_counts()
    .sort_index()
)

for scenario, count in (
    scenario_counts.items()
):

    percentage = (
        count
        / len(projects_df)
        * 100
    )

    print(
        f"  {scenario:<28}"
        f"{count:>4} "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# Stage distribution
# ============================================================

print()
print(
    "Stage distribution:"
)

stage_counts = (
    snapshots_df[
        "current_stage"
    ]
    .value_counts()
    .sort_index()
)

for stage, count in (
    stage_counts.items()
):

    percentage = (
        count
        / len(snapshots_df)
        * 100
    )

    print(
        f"  {stage:<5}"
        f"{count:>6} "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# Delay labels
# ============================================================

print()
print(
    "Forward delay label distribution:"
)

for column in [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]:

    positive = int(
        snapshots_df[
            column
        ].sum()
    )

    total = len(
        snapshots_df
    )

    percentage = (
        positive
        / total
        * 100
    )

    print(
        f"  {column:<18}"
        f"{positive:>6} positive "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# Label monotonicity
#
# A 30-day positive must also be positive at 60/90.
# A 60-day positive must also be positive at 90.
# ============================================================

print()
print(
    "Forward-label monotonicity:"
)

invalid_30_60 = (
    (
        snapshots_df[
            "delay_next_30d"
        ]
        == 1
    )
    &
    (
        snapshots_df[
            "delay_next_60d"
        ]
        == 0
    )
).sum()


invalid_60_90 = (
    (
        snapshots_df[
            "delay_next_60d"
        ]
        == 1
    )
    &
    (
        snapshots_df[
            "delay_next_90d"
        ]
        == 0
    )
).sum()


print(
    f"  30d=1 and 60d=0 : "
    f"{invalid_30_60}"
)

print(
    f"  60d=1 and 90d=0 : "
    f"{invalid_60_90}"
)

if (
    invalid_30_60 > 0
    or invalid_60_90 > 0
):

    raise ValueError(
        "Forward label monotonicity failed."
    )

print(
    "  Monotonicity: PASS"
)


# ============================================================
# Project outcome distribution
# ============================================================

print()
print(
    "Project outcome distribution:"
)

outcome_counts = (
    outcomes_df[
        "delayed"
    ]
    .value_counts()
    .sort_index()
)

for value, count in (
    outcome_counts.items()
):

    label = (
        "DELAYED"
        if value == 1
        else "NOT_DELAYED"
    )

    percentage = (
        count
        / len(outcomes_df)
        * 100
    )

    print(
        f"  {label:<15}"
        f"{count:>4} "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# Censoring
# ============================================================

print()
print(
    "Censored final projects:"
)

censored_count = int(
    outcomes_df[
        "is_censored"
    ].sum()
)

print(
    f"  {censored_count} / "
    f"{len(outcomes_df)}"
)


# ============================================================
# Temporal consistency
# ============================================================

print()
print("=" * 70)
print(
    "TEMPORAL CONSISTENCY CHECKS"
)
print("=" * 70)


# ------------------------------------------------------------
# Snapshot date should not precede project start.
# ------------------------------------------------------------

snap_dates = pd.to_datetime(
    snapshots_df[
        "snapshot_date"
    ]
)

project_dates = pd.to_datetime(
    snapshots_df[
        "project_start_date"
    ]
)

invalid_dates = (
    snap_dates
    < project_dates
).sum()

print(
    f"Snapshot before project start: "
    f"{invalid_dates}"
)

if invalid_dates > 0:

    raise ValueError(
        "Invalid snapshot dates detected."
    )


# ------------------------------------------------------------
# days_since_project_start
# ------------------------------------------------------------

calculated_days = (
    snap_dates
    - project_dates
).dt.days

stored_days = (
    snapshots_df[
        "days_since_project_start"
    ]
)

days_mismatch = (
    calculated_days
    != stored_days
).sum()

print(
    f"days_since_project_start "
    f"mismatches: {days_mismatch}"
)

if days_mismatch > 0:

    raise ValueError(
        "days_since_project_start "
        "is inconsistent."
    )


# ------------------------------------------------------------
# days_in_current_stage
# ------------------------------------------------------------

stage_entry_dates = pd.to_datetime(
    snapshots_df[
        "stage_entry_date"
    ]
)

calculated_stage_days = (
    snap_dates
    - stage_entry_dates
).dt.days

stored_stage_days = (
    snapshots_df[
        "days_in_current_stage"
    ]
)

stage_days_mismatch = (
    calculated_stage_days
    != stored_stage_days
).sum()

print(
    f"days_in_current_stage "
    f"mismatches: "
    f"{stage_days_mismatch}"
)

if stage_days_mismatch > 0:

    raise ValueError(
        "days_in_current_stage "
        "is inconsistent."
    )


print(
    "Temporal consistency: PASS"
)


# ============================================================
# Leakage checks
# ============================================================

print()
print("=" * 70)
print(
    "LEAKAGE CHECKS"
)
print("=" * 70)


FORBIDDEN_SNAPSHOT_COLUMNS = [
    "scenario",
    "scenario_ground_truth",
    "future_delay_day",
    "latent_expected_duration",
    "future_delay_probability",
    "synthetic_ground_truth_note",
]


leakage_columns = [
    column
    for column in FORBIDDEN_SNAPSHOT_COLUMNS
    if column in snapshots_df.columns
]


if leakage_columns:

    raise ValueError(
        "LEAKAGE DETECTED in snapshots: "
        + ", ".join(
            leakage_columns
        )
    )


print(
    "Future ground-truth variables: "
    "NOT PRESENT"
)

print(
    "Scenario ground truth in snapshots: "
    "NOT PRESENT"
)


# ------------------------------------------------------------
# Required labels
# ------------------------------------------------------------

required_labels = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]


missing_labels = [
    label
    for label in required_labels
    if label not in snapshots_df.columns
]


if missing_labels:

    raise ValueError(
        "Missing labels: "
        + ", ".join(
            missing_labels
        )
    )


print(
    "Forward labels: PASS"
)


# ============================================================
# Referential integrity
# ============================================================

print()
print(
    "Referential integrity:"
)

project_ids = set(
    projects_df[
        "project_id"
    ]
)

snapshot_project_ids = set(
    snapshots_df[
        "project_id"
    ]
)

outcome_project_ids = set(
    outcomes_df[
        "project_id"
    ]
)

if (
    project_ids
    != snapshot_project_ids
    or project_ids
    != outcome_project_ids
):

    raise ValueError(
        "Project ID mismatch across datasets."
    )

print(
    "  Projects ↔ snapshots ↔ outcomes: "
    "PASS"
)


# ============================================================
# Snapshot uniqueness
# ============================================================

duplicate_snapshots = (
    snapshots_df[
        "snapshot_id"
    ]
    .duplicated()
    .sum()
)

if duplicate_snapshots > 0:

    raise ValueError(
        "Duplicate snapshot_id detected."
    )

print(
    "  Snapshot uniqueness: PASS"
)


# ============================================================
# Project outcome consistency
# ============================================================

# A project marked delayed must have a first delay day.

invalid_outcomes = (
    (
        outcomes_df[
            "delayed"
        ] == 1
    )
    &
    (
        outcomes_df[
            "first_delay_day"
        ].isna()
    )
).sum()

if invalid_outcomes > 0:

    raise ValueError(
        "Delayed projects without "
        "first_delay_day detected."
    )

print(
    "  Outcome consistency: PASS"
)


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print(
    "OUTPUT FILES"
)
print("=" * 70)

print(
    f"Projects : {PROJECTS_FILE}"
)

print(
    f"Snapshots: {SNAPSHOTS_FILE}"
)

print(
    f"Outcomes : {OUTCOMES_FILE}"
)

print()
print("=" * 70)
print(
    "Synthetic temporal dataset generated "
    "and validated successfully."
)
print("=" * 70)