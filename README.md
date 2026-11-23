# Steel surface inspection

This project studies steel surface defect localization, a rare-defect learning
curve, and the trade-off between automatic alerts and human review. The data
preparation uses the Severstal steel defect dataset.

Use Python 3.13. From the repository root on Windows, create and activate the
project environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You can also run `.\.venv\Scripts\python.exe` directly without activation.
Install the project dependencies, then prepare the dataset using the
[DataPrep instructions](DataPrep/README.md):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements capture the observed Windows Python 3.13 environment, including
transitive packages. A clean installation has not yet been verified.

Double-click [start_mlflow.bat](start_mlflow.bat), or run it from the project
root, then open http://127.0.0.1:5000. Keep the window open while using MLflow
and press Ctrl+C to stop it. The interface compares recorded runs. Training
scripts launch experiments and choose their parameters. See the
[MLflow guide](projet/mlops/README.md) for integration and an example.

Reusable augmentation is under [preprocessing](projet/preprocessing/),
experiment tracking is under [mlops](projet/mlops/), future model training
belongs in [training](projet/training/), and checks are in [tests](projet/tests/).
Run the checks from the project root with
`.\.venv\Scripts\python.exe -m unittest discover -s projet/tests -t .`.
The [preprocessing guide](projet/preprocessing/README.md) describes the
manifest-based loader and train-only augmentation.
The [training guide](projet/training/README.md) describes the first binary
baseline experiments and their evaluation boundaries.

`DataPrep/` contains preparation code, the fixed split and the study evidence
log. The generated dataset lives under `donnees/severstal/` and is excluded from
Git. Final notebooks, saved outputs, model weights and the report will be placed
under `projet/` as the study progresses.
