# Retained validation baselines

These artifacts are experimental validation references. The final test remains
unopened. The datasets are acquired and prepared separately using the project
data-preparation guide. Install Git LFS and run `git lfs pull` after cloning.

| Directory | Contents | Purpose |
|---|---|---|
| `localization/` | Full-training classical detector, tuning results and fixed-threshold comparison predictions | Class-aware reference |
| `image-alert/` | Full-training Extra Trees model and validation results | Binary image-alert reference |
| `mobilenet-small/` | Frozen backbone, fitted head and validation scores | Lightweight CNN reference |
| `efficientnet-b2/` | Frozen backbone, fitted head and validation scores | EfficientNet CNN reference |

The class-aware model used multiscale proposals, texture features, Extra Trees
and threshold `-0.09999999999999998`. It matched only class 3 on the validation
comparison images. Its saved comparison report contains IDs, boxes and scores.
The training run is `a371f915bc8c487d86b01c34a564b731`, and the comparison run
is `3c9bd18408d34c3f8333fbea26f84534`. Evaluation conventions and limitations
are in `../../../DataPrep/report_evidence.md`.

SHA-256 hashes of the major weight files:

- Localization model: `e467c3aa9e391dc168e89872f39e629db6bf66a484b943df1d7570270a5ce8b3`
- Image-alert model: `d9e989dfbe52324888c6b16ca75d6f9e184a7999e149f09019b382666f39c003`
- MobileNetV3-Small backbone: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- EfficientNet-B2 backbone: `c35c147384e385a5bab5a8eabdabbe5a3df0487ee4a554108626ae474a5bf755`

The retained artifacts support replay using the project environment and prepared
data. Clean-clone installation and end-to-end replay have not yet been timed.
The local MLflow history is separate and its artifact paths are machine-specific.
Joblib model files should only be loaded from a trusted source.
