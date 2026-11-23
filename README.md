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
Install data-preparation dependencies and prepare the dataset using the
[DataPrep instructions](DataPrep/README.md). Install the detector's CPU
dependencies with `python -m pip install -r requirements-training.txt`.

`DataPrep/` contains preparation code, the fixed split and the study evidence
log. The generated dataset lives under `donnees/severstal/` and is excluded from
Git. Final notebooks, saved outputs, model weights and the report will be placed
under `projet/` as the study progresses.
