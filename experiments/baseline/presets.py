"""Train preset fragments owned by the baseline plan."""

from __future__ import annotations

from typing import Any

TRAIN_PRESET_ENTRIES: dict[str, dict[str, Any]] = {
    "plan09-composite": {
        "loss_system": "composite",
        "lambda_elev": 1.0,
        "lambda_slope": 0.5,
        "batch_size": 4,
        "workers": 1,
    },
}
