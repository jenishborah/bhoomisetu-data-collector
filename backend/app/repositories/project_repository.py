"""CSV-backed project repository for the BhoomiSetu prototype.

The synthetic snapshot table represents temporal state. The project master
table represents identity and geography. Keeping that join here means API
callers always receive one coherent project record without accidentally
counting every historical snapshot as a project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = PROJECT_ROOT / "output" / "synthetic"
SNAPSHOTS_PATH = SYNTHETIC_DIR / "synthetic_snapshots.csv"
PROJECTS_PATH = SYNTHETIC_DIR / "synthetic_projects.csv"

# Never expose the generator-only `scenario` field. It was deliberately not
# used by the model and must not be presented as a predictive signal.
MASTER_COLUMNS = [
    "project_id",
    "project_name",
    "project_type",
    "implementing_agency",
    "state",
    "district",
    "date_initiation",
]


def _json_value(value: Any) -> Any:
    """Turn pandas/numpy scalar values into values FastAPI can safely encode."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _record_to_json(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in record.items()}


class ProjectRepository:
    """In-memory, read-only view of local synthetic project data."""

    def __init__(self) -> None:
        self.data: pd.DataFrame | None = None
        self.master: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self) -> None:
        if not SNAPSHOTS_PATH.exists():
            raise FileNotFoundError(f"Synthetic snapshot dataset not found: {SNAPSHOTS_PATH}")
        if not PROJECTS_PATH.exists():
            raise FileNotFoundError(f"Synthetic project dataset not found: {PROJECTS_PATH}")

        snapshots = pd.read_csv(SNAPSHOTS_PATH)
        master = pd.read_csv(PROJECTS_PATH)
        if snapshots.empty or master.empty:
            raise ValueError("Synthetic project datasets must not be empty.")

        missing_snapshot = {"project_id", "snapshot_date"} - set(snapshots.columns)
        missing_master = set(MASTER_COLUMNS) - set(master.columns)
        if missing_snapshot:
            raise ValueError(f"Snapshot dataset is missing required columns: {sorted(missing_snapshot)}")
        if missing_master:
            raise ValueError(f"Project dataset is missing required columns: {sorted(missing_master)}")

        snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce")
        master = master[MASTER_COLUMNS].drop_duplicates("project_id")
        self.master = master
        self.data = snapshots.merge(master, on="project_id", how="left", validate="many_to_one")

    def _df(self) -> pd.DataFrame:
        if self.data is None:
            self._load_data()
        assert self.data is not None
        return self.data

    def _latest_df(self) -> pd.DataFrame:
        df = self._df()
        return (
            df.sort_values(["project_id", "snapshot_date"])
            .groupby("project_id", as_index=False)
            .tail(1)
            .sort_values("project_id")
            .copy()
        )

    def latest_project_frame(self) -> pd.DataFrame:
        """Return one latest temporal state per project for aggregations."""
        return self._latest_df()

    def list_projects(
        self,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
        state: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        latest = self._latest_df()
        if search:
            needle = search.strip().casefold()
            columns = ["project_id", "project_name", "district", "implementing_agency"]
            mask = pd.Series(False, index=latest.index)
            for column in columns:
                mask |= latest[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
            latest = latest[mask]
        if state:
            latest = latest[latest["state"].fillna("").str.casefold() == state.casefold()]
        if stage:
            latest = latest[latest["current_stage"].fillna("").str.casefold() == stage.casefold()]

        page = latest.iloc[offset : offset + limit]
        return [_record_to_json(item) for item in page.to_dict(orient="records")]

    def count_projects(
        self,
        search: str | None = None,
        state: str | None = None,
        stage: str | None = None,
    ) -> int:
        return len(self.list_projects(limit=10_000, search=search, state=state, stage=stage))

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        project = self._df()[self._df()["project_id"] == project_id].sort_values("snapshot_date")
        if project.empty:
            return None
        return _record_to_json(project.iloc[-1].to_dict())

    def get_project_snapshots(self, project_id: str) -> list[dict[str, Any]]:
        project = self._df()[self._df()["project_id"] == project_id].sort_values("snapshot_date")
        return [_record_to_json(item) for item in project.to_dict(orient="records")]

    def get_latest_snapshot(self, project_id: str) -> dict[str, Any] | None:
        return self.get_project(project_id)


_repository: ProjectRepository | None = None


def get_repository() -> ProjectRepository:
    global _repository
    if _repository is None:
        _repository = ProjectRepository()
    return _repository
