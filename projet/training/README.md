# Training

## Classical image-alert baselines

`baselines.py` compares full-image binary defect-alert methods. Both resize the
grayscale image to 256 by 64. One extracts HOG features and fits a standardized,
class-balanced stochastic linear SVM. The other combines spatial grayscale and gradient
features with an intensity histogram and fits a class-balanced random forest.
These produce image-level alert scores and have no defect type or location.

After installing the dependencies in the root README, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m projet.training.baselines --output .cache/binary-baselines-full
```

By default this uses the full fixed training and validation splits. Supply
`--train-count`, `--validation-count`, `--seed`, `--methods` or `--augment` to
change the experiment. Augmentation is off by default and only affects training.
Sample IDs are seeded and stored. Existing output directories are not overwritten.

Validation thresholds minimize a hypothetical image-level cost of 10 per
missed image and 1 per false alert. That is an image-level scenario distinct
from the box-level cost in the evaluation protocol. The threshold is selected
and reported on the same validation sample, so its metrics are optimistic for
generalization. AP and ROC-AUC use continuous scores. Image-level recall for a
class means alerting on images containing that class, including multi-class
images. It does not measure class identification.

MLflow and the output directory contain model files, validation scores and IDs,
metrics, fitted parameters and feature extraction code. Feature extraction time
includes data loading. The single-image timing uses the first 50 sampled
validation images after warm-up and measures decoded grayscale image through
feature extraction and prediction, excluding disk access. Full environment,
hardware and deployment-pipeline timing still need to be recorded when models
are compared for release. The test split is not opened.

### Classical tuning study

`python -m projet.training.tune_binary --output .cache/binary-tuning-v1`
extracts the original spatial features once, fits four tree-based configurations
on all training images, and splits validation into equally sized, binary-stratified
tuning and comparison portions with seed 0. Thresholds minimize the specified
10:1 image cost on the tuning portion. Report the other portion separately and
select the configuration from tuning evidence. The comparison portion is still
validation, and repeated examination can make it less independent. A final frozen
system must be measured once on test. Runs store split IDs, scores, settings,
metrics and models in local MLflow.
Use `--methods` to select configurations and `--comparison-fraction` to set
the validation partition. The proposal study accepts `--sigmas`, `--cutoffs`,
`--kernels`, `--min-areas` and `--max-regions` to change its grid.

`python -m projet.training.proposal_study --output .cache/proposal-study/results.json`
measures localization proposal recall at IoU 0.5 across a contrast grid on the
first 100 images of that validation tuning portion. This is a proposal-only
diagnostic. End-to-end defect detection must also classify the proposed regions
and be evaluated on shared validation images.

## Classical localization pilot

`region_baseline.py` fits a class-balanced HOG and linear-SVM crop classifier
for the four defect classes and background. At inference, connected components
of local image contrast produce proposals, the SVM assigns classes and scores,
and same-class NMS removes overlaps. Boxes are evaluated against the derived
component annotations. For example:

```powershell
.\.venv\Scripts\python.exe -m projet.training.region_baseline --train-count 500 --validation-count 100 --cutoff 25 --min-area 60 --output .cache/region-pilot
```

The run records proposals, predictions, class metrics, sampled IDs, fitted model
and source in MLflow. Its reported score threshold is tuned on the same validation
sample used for its reported metrics. Proposal recall is measured at IoU 0.5,
before classification. AP uses the retained top-100 proposals per image. Timing
starts from an already decoded image and includes proposals, crop features,
classification and NMS. Pilot timing includes the first invocation and is not a
release benchmark. The three initial proposal configurations performed poorly
on a 100-image pilot, so a stronger localization method is still needed.

The tuning study supports `--sigma`, `--kernel` and `--validation-ids` to
compare proposals on the same images. `--hard-negatives` adds candidate crops
matched to reference boxes and difficult background proposals to the original
ground-truth and random-background crops. `--classifier sgd` selects the
stochastic linear SVM. For a separate comparison partition, pass its IDs through
`--validation-ids` and `--validation-id-key`, and pass the threshold selected
on tuning with `--fixed-threshold`. Model and proposal settings must stay fixed.
