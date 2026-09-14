from pathlib import Path
import sys
import pandas as pd

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.predictor import get_predictor


DATA_PATH = (
    PROJECT_ROOT
    / "output"
    / "synthetic"
    / "synthetic_snapshots.csv"
)


def main():
    print("=" * 70)
    print("BhoomiSetu Backend Predictor Test")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Snapshots loaded : {len(df):,}")
    print(f"Columns          : {len(df.columns)}")

    # Use the first snapshot as a real test record.
    row = df.iloc[0].to_dict()

    print()
    print("Test snapshot")
    print("-" * 70)

    for key in [
        "project_id",
        "snapshot_id",
        "snapshot_date",
        "current_stage",
    ]:
        if key in row:
            print(f"{key:25s}: {row[key]}")

    predictor = get_predictor()

    result = predictor.predict(row)

    print()
    print("Prediction")
    print("-" * 70)

    for horizon, values in result["predictions"].items():
        print(
            f"{horizon:5s} | "
            f"Raw: {values['raw_probability'] * 100:7.3f}% | "
            f"Calibrated: "
            f"{values['calibrated_probability'] * 100:7.3f}%"
        )

    print()
    print(f"Model version: {result['model_version']}")

    print()
    print("=" * 70)
    print("PREDICTOR TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()