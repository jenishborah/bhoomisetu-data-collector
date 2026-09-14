from __future__ import annotations

from typing import Any


# ================================================================
# BhoomiSetu Risk Engine
# ================================================================
# Prototype UI risk bands.
#
# IMPORTANT:
# These are NOT statutory, legal, or government-approved thresholds.
# They are only used to make the prototype dashboard understandable.
# ================================================================


# The calibrated synthetic model estimates a relatively rare event.  These
# deliberately compact presentation thresholds retain useful separation in a
# prototype dashboard; they are not legal, statutory, or operating limits.
RISK_BANDS = [
    (0.20, "LOW"),
    (0.40, "MODERATE"),
    (0.70, "HIGH"),
    (1.01, "CRITICAL"),
]


def risk_band(
    probability: float,
) -> str:
    """
    Convert a probability in [0, 1] into a prototype risk band.
    """

    p = max(
        0.0,
        min(1.0, float(probability)),
    )

    for upper_limit, label in RISK_BANDS:
        if p < upper_limit:
            return label

    return "CRITICAL"


def risk_score(
    probability: float,
) -> int:
    """
    Convert probability to a simple 0-100 dashboard score.
    """

    p = max(
        0.0,
        min(1.0, float(probability)),
    )

    return round(p * 100)


def _trend(
    probabilities: list[float],
) -> str:
    """
    Estimate the direction of the risk curve from 30d to 90d.

    Prototype interpretation:
      >= +20 percentage points -> RISING
      <= -20 percentage points -> FALLING
      otherwise                -> STABLE
    """

    if len(probabilities) < 2:
        return "UNKNOWN"

    delta = (
        probabilities[-1]
        - probabilities[0]
    )

    if delta >= 0.20:
        return "RISING"

    if delta <= -0.20:
        return "FALLING"

    return "STABLE"


def build_risk_summary(
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert predictor output into a dashboard-friendly object.

    Expected:
      {
        "30d": {
            "raw_probability": 0.1,
            "calibrated_probability": 0.08
        },
        ...
      }
    """

    horizons: dict[str, dict[str, Any]] = {}

    for horizon in [
        "30d",
        "60d",
        "90d",
    ]:

        item = predictions.get(horizon)

        if item is None:
            continue

        probability = float(
            item["calibrated_probability"]
        )

        horizons[horizon] = {
            "probability": probability,
            "percentage": round(
                probability * 100,
                1,
            ),
            "score": risk_score(
                probability
            ),
            "band": risk_band(
                probability
            ),
        }

    if not horizons:
        raise ValueError(
            "No horizon predictions supplied."
        )

    # For the main project status, use 90-day risk because it gives
    # the broadest early-warning view.
    headline = horizons.get(
        "90d"
    )

    if headline is None:
        headline = next(
            iter(horizons.values())
        )

    probabilities = [
        horizons[h]["probability"]
        for h in [
            "30d",
            "60d",
            "90d",
        ]
        if h in horizons
    ]

    return {
        "headline_band": headline["band"],
        "headline_score": headline["score"],
        "headline_percentage": headline["percentage"],
        "trend": _trend(probabilities),
        "horizons": horizons,
        "thresholds": {
            "low": "<1%",
            "moderate": "1-2.9%",
            "high": "3-9.9%",
            "critical": "10%+",
            "note": (
                "Prototype risk bands only; "
                "not statutory thresholds."
            ),
        },
    }


def build_project_risk_response(
    prediction_response: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine predictor output with the risk interpretation layer.
    """

    predictions = prediction_response.get(
        "predictions"
    )

    if not predictions:
        raise ValueError(
            "Prediction response contains no predictions."
        )

    return {
        "risk": build_risk_summary(
            predictions
        ),
        "model_version": prediction_response.get(
            "model_version"
        ),
    }
