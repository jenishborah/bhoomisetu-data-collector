from pathlib import Path
import csv
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "54635"

DATA_DIR = (
    Path("output")
    / "normalized"
    / "bhoomirashi"
    / PROJECT_ID
)

NOTIFICATIONS_FILE = DATA_DIR / "notifications.csv"
SANCTIONS_FILE = DATA_DIR / "sanctions.csv"
OUTPUT_FILE = DATA_DIR / "stage_timeline.csv"


# ============================================================
# HELPERS
# ============================================================

def parse_date(value):
    if not value:
        return None

    return datetime.strptime(
        value,
        "%Y-%m-%d"
    ).date()


def days_between(date_a, date_b):
    if date_a is None or date_b is None:
        return None

    return (date_b - date_a).days


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


# ============================================================
# LOAD NOTIFICATIONS
# ============================================================

def load_notifications():

    rows = load_csv(
        NOTIFICATIONS_FILE
    )

    notifications = []

    for row in rows:

        notifications.append({
            "project_id": row["project_id"],
            "notification_id": row["notification_id"],
            "notification_type": row["notification_type"],
            "publish_date": parse_date(
                row["publish_date"]
            ),
            "notification_number": row[
                "notification_number"
            ],
            "status": row["status"],
        })

    notifications.sort(
        key=lambda x: x["publish_date"]
    )

    return notifications


# ============================================================
# LOAD SANCTIONS
# ============================================================

def load_sanctions():

    rows = load_csv(
        SANCTIONS_FILE
    )

    sanctions = []

    for row in rows:

        sanctions.append({
            "project_id": row["project_id"],
            "sanction_number": row[
                "sanction_number"
            ],
            "sanction_date": parse_date(
                row["sanction_date"]
            ),
            "sanction_amount_rs": float(
                row["sanction_amount_rs"]
            ),
        })

    sanctions.sort(
        key=lambda x: x["sanction_date"]
    )

    return sanctions


# ============================================================
# BUILD TIMELINE
# ============================================================

def build_timeline():

    notifications = load_notifications()
    sanctions = load_sanctions()

    events = []

    # --------------------------------------------------------
    # Sanction events
    # --------------------------------------------------------

    for sanction in sanctions:

        events.append({
            "event_date": sanction["sanction_date"],
            "event_type": "SANCTION",
            "stage": "PROJECT_SANCTION",
            "event_id": sanction["sanction_number"],
            "notification_number": "",
            "status": "SANCTIONED",
            "amount_rs": sanction["sanction_amount_rs"],
            "days_since_previous_event": None,
        })

    # --------------------------------------------------------
    # Notification stage mapping
    # --------------------------------------------------------

    stage_mapping = {
        "3a": "3A_PRELIMINARY_NOTIFICATION",
        "3A": "3A_DECLARATION",
        "3D": "3D_DECLARATION",
    }

    # --------------------------------------------------------
    # Notification events
    # --------------------------------------------------------

    for notification in notifications:

        notification_type = notification[
            "notification_type"
        ]

        stage = stage_mapping.get(
            notification_type,
            notification_type,
        )

        event = {
            "event_date": notification[
                "publish_date"
            ],
            "event_type": "NOTIFICATION",
            "stage": stage,
            "event_id": notification[
                "notification_id"
            ],
            "notification_number": notification[
                "notification_number"
            ],
            "status": notification[
                "status"
            ],
            "amount_rs": None,
            "days_since_previous_event": None,
        }

        events.append(event)

    # --------------------------------------------------------
    # Sort complete timeline
    # --------------------------------------------------------

    events.sort(
        key=lambda x: (
            x["event_date"],
            x["event_type"],
            str(x["event_id"]),
        )
    )

    # --------------------------------------------------------
    # Calculate gaps
    # --------------------------------------------------------

    previous_date = None

    for event in events:

        event["days_since_previous_event"] = (
            days_between(
                previous_date,
                event["event_date"]
            )
        )

        previous_date = event["event_date"]

    return events


# ============================================================
# SAVE TIMELINE
# ============================================================

def save_timeline(events):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "project_id",
        "event_sequence",
        "event_date",
        "event_type",
        "stage",
        "event_id",
        "notification_number",
        "status",
        "amount_rs",
        "days_since_previous_event",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for sequence, event in enumerate(
            events,
            start=1
        ):

            row = {
                "project_id": PROJECT_ID,
                "event_sequence": sequence,
                "event_date": event[
                    "event_date"
                ].isoformat(),
                "event_type": event[
                    "event_type"
                ],
                "stage": event[
                    "stage"
                ],
                "event_id": event[
                    "event_id"
                ],
                "notification_number": event[
                    "notification_number"
                ],
                "status": event[
                    "status"
                ],
                "amount_rs": (
                    event["amount_rs"]
                    if event["amount_rs"] is not None
                    else ""
                ),
                "days_since_previous_event": event[
                    "days_since_previous_event"
                ],
            }

            writer.writerow(row)

    return OUTPUT_FILE


# ============================================================
# SUMMARY
# ============================================================

def print_summary(events):

    print()
    print("=" * 70)
    print("BHOOMISETU STAGE TIMELINE")
    print("=" * 70)

    for event in events:

        amount = ""

        if event["amount_rs"] is not None:
            amount = (
                f" | ₹{event['amount_rs']:,.2f}"
            )

        gap = ""

        if (
            event["days_since_previous_event"]
            is not None
        ):
            gap = (
                f" | +{event['days_since_previous_event']} days"
            )

        print(
            f"{event['event_date']} | "
            f"{event['event_type']} | "
            f"{event['stage']} | "
            f"{event['event_id']} | "
            f"{event['notification_number']} | "
            f"{event['status']}"
            f"{amount}{gap}"
        )

    print()
    print(
        f"Total timeline events: {len(events)}"
    )

    if events:

        first_date = events[0]["event_date"]
        last_date = events[-1]["event_date"]

        span = days_between(
            first_date,
            last_date
        )

        print(
            f"Timeline span: {span} days"
        )

    print()
    print(f"Saved: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BhoomiSetu Stage Timeline Builder")
    print("=" * 70)

    print(
        f"Project ID: {PROJECT_ID}"
    )

    notifications = load_notifications()
    sanctions = load_sanctions()

    print(
        f"Notifications loaded: "
        f"{len(notifications)}"
    )

    print(
        f"Sanctions loaded: "
        f"{len(sanctions)}"
    )

    events = build_timeline()

    save_timeline(events)

    print_summary(events)


if __name__ == "__main__":
    main()