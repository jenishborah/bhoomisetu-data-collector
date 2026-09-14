from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
SHAP_DIR = ROOT / "output" / "synthetic" / "ml" / "shap"
OUT_DIR = SHAP_DIR / "dependence_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "days_in_current_stage",
    "days_since_project_start",
    "docs_complete_pct",
    "disbursal_pct",
    "grievance_redressal_days",
    "file_pending_days",
    "case_age_months",
    "land_acquisition_ratio",
    "rnr_progress_pct",
    "grievances_30d",
]

TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]

LOCAL_PATH = SHAP_DIR / "local_shap_explanations.csv"

if not LOCAL_PATH.exists():
    raise FileNotFoundError(
        f"Missing SHAP file: {LOCAL_PATH}"
    )

local = pd.read_csv(LOCAL_PATH)

required_columns = [
    "target",
    "project_id",
    "snapshot_id",
    "snapshot_date",
    "current_stage",
    "feature",
    "feature_value",
    "shap_value",
    "direction",
]

missing = [
    c for c in required_columns
    if c not in local.columns
]

if missing:
    raise ValueError(
        f"SHAP file is missing required columns: {missing}"
    )

print("=" * 72)
print("BhoomiSetu SHAP Dependence / Sanity Audit")
print("=" * 72)
print(f"SHAP rows   : {len(local):,}")
print(f"Projects    : {local['project_id'].nunique():,}")
print(f"Snapshots   : {local['snapshot_id'].nunique():,}")

# ---------------------------------------------------------------------
# Dependence summary
# ---------------------------------------------------------------------

def summarize_dependence(df):

    d = df[
        ["feature_value", "shap_value"]
    ].copy()

    d = d.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(d) < 30:
        return {
            "n": len(d),
            "trend": "insufficient_data",
            "min_value": np.nan,
            "max_value": np.nan,
            "mean_shap": np.nan,
            "min_shap": np.nan,
            "max_shap": np.nan,
        }

    if d["feature_value"].nunique() < 4:
        return {
            "n": len(d),
            "trend": "low_variation",
            "min_value": float(d["feature_value"].min()),
            "max_value": float(d["feature_value"].max()),
            "mean_shap": float(d["shap_value"].mean()),
            "min_shap": float(d["shap_value"].min()),
            "max_shap": float(d["shap_value"].max()),
        }

    try:
        d["bin"] = pd.qcut(
            d["feature_value"],
            q=10,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        d["bin"] = pd.cut(
            d["feature_value"],
            bins=10,
            labels=False,
            duplicates="drop",
        )

    g = (
        d.groupby("bin", observed=True)
        .agg(
            n=("shap_value", "size"),
            value_median=("feature_value", "median"),
            mean_shap=("shap_value", "mean"),
        )
        .reset_index()
    )

    diffs = np.diff(
        g["mean_shap"].values
    )

    if len(diffs) == 0:
        trend = "insufficient_data"
    else:

        positive = np.mean(diffs > 0)
        negative = np.mean(diffs < 0)

        if positive >= 0.70:
            trend = "mostly_increasing_risk"

        elif negative >= 0.70:
            trend = "mostly_decreasing_risk"

        else:
            trend = "nonlinear_or_mixed"

    return {
        "n": len(d),
        "trend": trend,
        "min_value": float(d["feature_value"].min()),
        "max_value": float(d["feature_value"].max()),
        "mean_shap": float(d["shap_value"].mean()),
        "min_shap": float(d["shap_value"].min()),
        "max_shap": float(d["shap_value"].max()),
    }


summaries = []

# ---------------------------------------------------------------------
# Analyze every target × feature
# ---------------------------------------------------------------------

for target in TARGETS:

    target_data = local[
        local["target"] == target
    ]

    print("")
    print(f"TARGET: {target}")
    print("-" * 72)

    for feature in FEATURES:

        sub = target_data[
            target_data["feature"] == feature
        ].copy()

        if sub.empty:
            print(
                f"{feature}: NOT FOUND"
            )
            continue

        result = summarize_dependence(
            sub
        )

        result.update({
            "target": target,
            "feature": feature,
        })

        summaries.append(result)

        print(
            f"{feature:35s} "
            f"{result['trend']:28s} "
            f"n={result['n']}"
        )

        # -------------------------------------------------------------
        # Decile table
        # -------------------------------------------------------------

        d = sub[
            ["feature_value", "shap_value"]
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if (
            len(d) >= 20
            and d["feature_value"].nunique() >= 4
        ):

            try:
                d["decile"] = (
                    pd.qcut(
                        d["feature_value"],
                        q=10,
                        labels=False,
                        duplicates="drop",
                    ) + 1
                )
            except Exception:
                d["decile"] = (
                    pd.cut(
                        d["feature_value"],
                        bins=10,
                        labels=False,
                        duplicates="drop",
                    ) + 1
                )

            deciles = (
                d.groupby(
                    "decile",
                    observed=True
                )
                .agg(
                    n=("shap_value", "size"),
                    value_min=("feature_value", "min"),
                    value_median=("feature_value", "median"),
                    value_max=("feature_value", "max"),
                    mean_shap=("shap_value", "mean"),
                    median_shap=("shap_value", "median"),
                )
                .reset_index()
            )

            deciles.insert(
                0,
                "target",
                target
            )

            deciles.insert(
                1,
                "feature",
                feature
            )

            deciles.to_csv(
                OUT_DIR
                / f"{target}__{feature}__deciles.csv",
                index=False
            )


summary_df = pd.DataFrame(
    summaries
)

summary_df = summary_df.sort_values(
    ["target", "feature"]
)

summary_path = (
    OUT_DIR
    / "dependence_audit_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# ---------------------------------------------------------------------
# Aggregate stage SHAP
# ---------------------------------------------------------------------

stage_features = [
    f
    for f in local["feature"].dropna().unique()
    if str(f).startswith("current_stage_")
]

stage_summary_path = (
    OUT_DIR
    / "stage_aggregated_shap.csv"
)

if stage_features:

    stage_data = local[
        local["feature"].isin(
            stage_features
        )
    ].copy()

    stage_agg = (
        stage_data.groupby(
            [
                "target",
                "project_id",
                "snapshot_id",
                "current_stage",
            ],
            dropna=False,
        )["shap_value"]
        .sum()
        .reset_index(
            name="stage_shap_value"
        )
    )

    stage_summary = (
        stage_agg
        .groupby(
            [
                "target",
                "current_stage",
            ]
        )
        .agg(
            n=("stage_shap_value", "size"),
            mean_shap=(
                "stage_shap_value",
                "mean",
            ),
            median_shap=(
                "stage_shap_value",
                "median",
            ),
            min_shap=(
                "stage_shap_value",
                "min",
            ),
            max_shap=(
                "stage_shap_value",
                "max",
            ),
        )
        .reset_index()
    )

    stage_summary.to_csv(
        stage_summary_path,
        index=False
    )

# ---------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------

lines = [
    "BhoomiSetu SHAP Dependence / Sanity Audit",
    "=" * 72,
    "",
    "Data source:",
    "  local_shap_explanations.csv",
    "",
    "Interpretation:",
    "  SHAP values describe model attribution, not causality.",
    "  Results are based on synthetic prototype data.",
    "",
]

for target in TARGETS:

    lines.append(
        f"TARGET: {target}"
    )

    lines.append(
        "-" * 72
    )

    t = summary_df[
        summary_df["target"] == target
    ]

    for _, row in t.iterrows():

        lines.append(
            f"{row['feature']}: "
            f"{row['trend']} | "
            f"n={int(row['n'])} | "
            f"range="
            f"{row['min_value']:.4g}"
            f".."
            f"{row['max_value']:.4g} | "
            f"SHAP="
            f"{row['min_shap']:.4f}"
            f".."
            f"{row['max_shap']:.4f} | "
            f"mean="
            f"{row['mean_shap']:.4f}"
        )

    lines.append("")

report_path = (
    OUT_DIR
    / "dependence_audit_report.txt"
)

report_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("")
print("=" * 72)
print("SHAP DEPENDENCE AUDIT COMPLETE")
print("=" * 72)
print(
    f"Summary : {summary_path}"
)
print(
    f"Report  : {report_path}"
)

if stage_summary_path.exists():
    print(
        f"Stage   : {stage_summary_path}"
    )

print("")
print(
    "PASS: Audit completed using feature_value "
    "already stored in the SHAP output."
)
print(
    "No model or dataset was modified."
)