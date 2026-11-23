# Local experiment tracking

Run `start_mlflow.bat` from the project root to view experiments at
http://127.0.0.1:5000. The database and artifacts are stored in
`../../.cache/mlflow/`, which Git ignores. Back up that directory separately to
keep run history. The previous `severstal-classical` pilot remains available in
this local store and is marked as a retrospective import.

## Sharing and reproducibility status

The SQLite database and artifact directory both need to be preserved for another
person to inspect run metrics and artifacts. On 2026-11-23 the local database
contained 41 run records and the artifact directory held approximately 415 MiB.
These figures change as experiments are added. Git ignores this store. Before a
backup, stop training and the MLflow server, then preserve the complete
`.cache/mlflow/` directory. Inspect artifacts for private paths or credentials
before sharing. Use a private storage location agreed with the recipient.

Artifact locations in the current database contain absolute paths to the
original machine. Copying the directory to a different project location alone
will not make the artifacts portable. A tested export/import or location
migration is still needed before delivering a copy to the professor. The
SQLite database can retain history independently of those artifact paths.

`../../requirements.txt` records `pip freeze` from the Windows Python 3.13
experiment environment on 2026-11-23, including transitive and auxiliary
packages. The CPU PyTorch index is included. This snapshot has not been verified
as a clean installation or a cross-platform lockfile.
Rebuilding the data also requires the Kaggle source and tracked split manifests;
see `../../DataPrep/README.md`. The clean-clone installation, artifact migration
and full replay have not yet been verified. Package versions, preprocessing,
split IDs, saved weights, model settings and environment must all be preserved
to attempt replay, and numerical identity across hardware is not guaranteed.

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
