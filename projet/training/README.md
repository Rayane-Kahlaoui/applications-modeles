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
the validation partition. `--features texture-multiscale` adds local mean and
variation in intensity, gradient and contrast at two spatial scales, resized to
512 by 82. The proposal study accepts `--sigmas`, `--cutoffs`,
`--kernels`, `--min-areas` and `--max-regions` to change its grid.
`--methods forest-cv` searches eight Random Forest configurations with three
training-only stratified folds, selecting by average precision before applying
the existing validation threshold and comparison procedure.

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

## Pretrained detector pilot

`detector.py` runs an optional Ultralytics YOLO11n detection pilot from a local
pretrained checkpoint. The initial weights came from the official Ultralytics
assets release. Record the source and SHA-256. Ultralytics and its model weights
are offered under AGPL-3.0 and enterprise licenses; assess the applicable license
before distribution or deployment.

```powershell
.\.venv\Scripts\python.exe -m projet.training.detector --weights .cache/weights/yolo11n.pt --validation-ids .cache/proposal-study-v2/results.json --epochs 5 --output .cache/yolo11-pilot
```

The pilot selects training images using seed 0 and takes validation IDs from the
supplied proposal-study result. It writes a local dataset configuration containing
only training and validation paths, trains on CPU at 320 pixels, then evaluates
the saved checkpoint using the shared box evaluator. Ultralytics training defaults
and its own transforms apply; the project augmentation helper is not used. MLflow
stores the trainer settings, source, checkpoint, predictions, IDs and results.
Threshold and best-checkpoint selection use the supplied tuning images. To run a
fixed checkpoint on a separate comparison list, use `--evaluate-only`,
`--validation-id-key comparison_ids` and `--fixed-threshold` selected beforehand.
This is an exploratory pilot and its local output and checkpoint are Git-ignored.

## Pretrained image classifiers

`cnn_classifier.py` compares frozen ImageNet-pretrained MobileNetV3-Small and
EfficientNet-B0 feature extractors with a balanced logistic-regression head.
It resizes the full grayscale image to 256 by 64, repeats it into three channels
and applies ImageNet normalization. This uses the whole steel image but differs
from the original ImageNet crop. The feature backbone is frozen and the head is
fitted using training images. It uses the same seeded 626/626 validation split
and tunes the 10:1 image-alert threshold only on the tuning half.

The default weight directory must contain the official Torchvision files
`mobilenet_v3_small-047dcff4.pth` and `efficientnet_b0_rwightman-7f5810bc.pth`.
They are available from the official Torchvision weight URLs
https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth and
https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth.
The run records the hash, backbone, source script, classifier head, sample IDs
and scores in MLflow. To replay, retain or obtain the same pretrained weight
file and check its hash. A run is launched with:

```powershell
.\.venv\Scripts\python.exe -m projet.training.cnn_classifier --model efficientnet-b0 --output .cache/efficientnet-binary
```

These classifiers detect whether an image has any annotated defect. They do not
identify defect type or location. Feature extraction duration includes image and
annotation loading and is not an end-to-end deployment latency benchmark.
With `--benchmark-only`, the same command measures the first 100 comparison
images after five warm-up predictions, with four CPU threads by default. It
times an already decoded image through resizing, feature extraction and head
prediction. To benchmark the saved forest, use `--model random-forest` and its
run output directory. Disk access is excluded and a clean-system benchmark is
still needed for deployment. The test split remains unopened.

## Custom CNN from scratch

`python -m projet.training.cnn_classifier --model custom-cnn --epochs 8 --output .cache/custom-cnn`
trains a compact four-block grayscale CNN initialized randomly. The script uses
90% of the training partition for gradient updates and 10% for checkpoint
selection by average precision, then uses the established validation tuning and
comparison partitions. These training holdout images do not enter gradient
updates. Train-time image augmentations include small rotations, horizontal
flips, brightness changes, noise and optional blur. The image-level label is
kept, and rotation is skipped for a positive image if a visible downsampled
mask cannot be retained. Small defects can disappear during resizing.

The run evaluates darker and brighter images, Gaussian noise and blur on the
comparison portion with the original selected threshold. Conditions are
synthetic and applied after resizing; they are not a factory-calibrated stress
test. Runs record their IDs, metrics, checkpoint, settings and source in MLflow.
The optional `--benchmark-only` workflow supports the saved custom model too.
The observed eight-epoch run took about 14 minutes for data caching and training
alone. Complete clean-clone replay time remains unmeasured; saved weights and
results can avoid retraining during replay.
