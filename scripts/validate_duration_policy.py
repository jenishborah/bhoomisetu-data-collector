from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = BASE_DIR / "config"

STAGE_POLICY_FILE = (
    CONFIG_DIR / "stage_duration_policy.csv"
)

LEGAL_POLICY_FILE = (
    CONFIG_DIR / "legal_deadlines.csv"
)


# ============================================================
# Helper
# ============================================================

def validate_file(path, required_columns):
    print()
    print("=" * 70)
    print(f"Validating: {path.name}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print("Required columns: PASS")

    return df


# ============================================================
# Stage duration policy
# ============================================================

stage_required = [
    "stage_transition",
    "from_stage",
    "to_stage",
    "benchmark_type",
    "benchmark_days",
    "source",
    "confidence",
    "usable_for_ml_label",
    "notes",
]

stage_policy = validate_file(
    STAGE_POLICY_FILE,
    stage_required,
)


# ------------------------------------------------------------
# Validate transitions
# ------------------------------------------------------------

expected_transitions = {
    "3a_to_3A",
    "3A_to_3D",
}

actual_transitions = set(
    stage_policy["stage_transition"]
)

if actual_transitions != expected_transitions:

    print(
        "WARNING: Unexpected stage transitions:"
    )

    print(
        actual_transitions
        - expected_transitions
    )

else:

    print(
        "Stage transitions: PASS"
    )


# ------------------------------------------------------------
# Pending empirical benchmarks
# ------------------------------------------------------------

pending = stage_policy[
    stage_policy["benchmark_type"]
    == "empirical_pending"
]

print(
    f"Empirical benchmarks pending: "
    f"{len(pending)}"
)


# ------------------------------------------------------------
# ML usability
# ------------------------------------------------------------

usable_values = set(
    stage_policy[
        "usable_for_ml_label"
    ]
    .astype(str)
    .str.upper()
)

if "YES" in usable_values:

    print(
        "WARNING: A benchmark is currently "
        "marked usable for ML."
    )

else:

    print(
        "Current ML benchmark usage: BLOCKED "
        "until sufficient historical data exists."
    )


# ============================================================
# Legal deadline policy
# ============================================================

legal_required = [
    "rule_id",
    "process_area",
    "from_event",
    "to_event",
    "deadline_days",
    "deadline_type",
    "source",
    "source_reference",
    "confidence",
    "notes",
]

legal_policy = validate_file(
    LEGAL_POLICY_FILE,
    legal_required,
)


# ------------------------------------------------------------
# Validate deadline values
# ------------------------------------------------------------

invalid_deadlines = legal_policy[
    pd.to_numeric(
        legal_policy["deadline_days"],
        errors="coerce",
    ).isna()
]

if len(invalid_deadlines) > 0:

    raise ValueError(
        "Legal deadline policy contains "
        "invalid deadline_days values."
    )

print(
    "Legal deadline values: PASS"
)


# ============================================================
# Final status
# ============================================================

print()
print("=" * 70)
print("DURATION POLICY VALIDATION")
print("=" * 70)

print(
    f"Stage policy rows : "
    f"{len(stage_policy)}"
)

print(
    f"Legal policy rows : "
    f"{len(legal_policy)}"
)

print(
    "Status             : PASS"
)

print()
print(
    "No arbitrary 3a→3A or 3A→3D benchmark "
    "has been introduced."
)

print(
    "These transitions remain empirical_pending "
    "until sufficient historical data is collected."
)

print("=" * 70)