# Local experiment tracking

Run `start_mlflow.bat` from the project root to view experiments at
http://127.0.0.1:5000. The database and artifacts are stored in
`../../.cache/mlflow/`, which Git ignores. Back up that directory separately to
keep run history. The previous `severstal-classical` pilot remains available in
this local store and is marked as a retrospective import.

Any training script run from the project root can use MLflow directly:

```python
import mlflow
from projet.mlops.tracking import start_run

with start_run("model-name", phase="validation", seed=0,
               description="Short description of this comparison"):
    mlflow.log_params({"learning_rate": 0.001})
    # Train and evaluate on the permitted split here.
    mlflow.log_metric("validation_metric", 0.5)  # Replace with the measured value.
    # mlflow.log_artifact("path/to/saved/model")
```

Runs are grouped under `severstal-inspection` by default and named using the
model, phase and seed. MLflow assigns a unique ID even when names repeat. The
dataset, model and phase are also searchable tags. Supply `experiment_name` to
`start_run` when a separate group is useful. The existing retrospective pilot
remains in `severstal-classical`.

In the Runs table, select runs to compare their parameters and metrics, use the
search box to filter runs, and open a run to inspect its artifacts and notes.
The old pilot grid runs are named with their recorded C, contrast and area
settings. Their evidence and retrospective provenance are unchanged.

Each model's training script must log its actual parameters, validation metrics,
dataset and split identifiers, and saved model artifacts. The helper configures
the local store. It does not automatically instrument future training scripts.
MLflow's UI compares runs. The local Windows server reports that MLflow job
execution is unsupported, so launch training scripts separately.
