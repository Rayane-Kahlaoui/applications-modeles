"""Configure local MLflow tracking for any project experiment."""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DIR = PROJECT_ROOT / ".cache" / "mlflow"
DEFAULT_EXPERIMENT = "severstal-inspection"


def configure_tracking(experiment_name=DEFAULT_EXPERIMENT):
    """Select an experiment with local SQLite metadata and local artifacts."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=(TRACKING_DIR / "artifacts").as_uri())
    mlflow.set_experiment(experiment_name)


def start_run(model, *, phase, seed, description=None, experiment_name=DEFAULT_EXPERIMENT):
    """Start a named Severstal run, ready to use as a context manager."""
    if not model.strip() or not phase.strip():
        raise ValueError("Model and phase must be nonempty")
    configure_tracking(experiment_name)
    run_name = f"{model.strip()} | {phase.strip()} | seed {seed}"
    tags = {"dataset": "Severstal", "model": model.strip(), "phase": phase.strip()}
    if description:
        tags["mlflow.note.content"] = description
    run = mlflow.start_run(run_name=run_name, tags=tags)
    mlflow.log_param("seed", seed)
    return run
