# Report evidence log

This log records decisions, checks, measured results, limitations and artifact
locations for the owner to use when writing the final report. Update it as the
study progresses. Proposed choices and unverified assumptions are labelled.

## Dataset and preparation

- **Source:** Severstal Steel Defect Detection competition on Kaggle,
  https://www.kaggle.com/competitions/severstal-steel-defect-detection/data.
  The owner selected this dataset for the study because it includes annotated
  defect images and images without annotated defects.
- **Observed inventory:** The supplied labelled training set contains 12,568
  images. All decoded successfully as RGB at 1,600 × 256 pixels. Its CSV has
  7,095 nonempty class masks. The image counts per class are 897 for class 1,
  247 for class 2, 5,150 for class 3 and 801 for class 4. Classes can co-occur.
  There are 427 images with multiple classes and 5,902 images without annotated
  defects. The distribution is a property of this dataset, not a measured
  production-line prevalence.
- **Identity:** The SHA-256 of `raw/train.csv` is
  `653e98583cdcacc71c74924e40baa6edd24b3710581742c1ddfddc0037c41c4e`.
  The original ZIP hash has not been recorded.
- **Integrity checks:** `inspect_data.py` checked decoding, dimensions,
  annotation IDs, class IDs, duplicate image/class rows, RLE structure and bounds,
  file hashes and decoded-pixel hashes. It reported no errors and found no
  byte-identical or decoded-pixel-identical training-image groups. Near-duplicate
  images have not been ruled out.
- **Approved split:** The owner approved the image-level split. `manifests/split.json` records seed 0 and an image-level
  split stratified by exact class set. The train, validation and test partitions
  contain 10,050, 1,252 and 1,266 images respectively. Disjointness was checked,
  and regenerating the manifest from the source annotations produced the same
  manifest.
  Production-run or coil identifiers are absent from the supplied metadata, so
  independence between related images across partitions is unverified. The
  competition's original unlabelled test images are outside this split.
- **Approved rare class:** The owner approved class 2. It has 196 images in training, 23 in
  validation and 28 in test. The manifest selects a seeded ordering of 15
  training images and nested subsets of 5, 10 and 15. Each condition retains
  all training images without class 2 and excludes other images containing it,
  including mixed-class images. The subset and exclusion checks passed. Class 2
  uses the dataset's numeric ID; its physical defect name is unverified.
- **Mask-to-box convention:** `masks_to_boxes.py` decodes one-indexed,
  column-major RLE into class masks, then draws one box around each 8-connected
  component. YOLO labels use normalized original-image coordinates and class
  IDs 0–3. There is a label file for each image, including empty files for
  images with no annotated defects. A component can be a fragment of one
  physical defect, and the resulting boxes are derived annotations.
- **Conversion checks:** Small RLE fixtures passed. All 12,568 generated label
  files passed class and coordinate-bound checks. A visual overlay was reviewed
  for one training image per class, saved locally as
  `processed/box_examples.png`. That check is a limited sample and the image is
  in an ignored directory. Broader annotation-quality assessment is pending.
- **Training-mask fragmentation check:** Counting 8-connected components per
  annotated training image and class found medians of 2, 1, 2 and 2 for classes
  1–4, with 90th percentiles of 8, 2, 6 and 5. The maxima were 20, 3, 19 and
  14. The numbers of masks with 10 or more components were 41, 0, 63 and 4.
  Six selected training-image overlays are in the ignored file
  `processed/fragmentation_examples.png`. Multiple components can represent
  distinct regions or parts of one physical defect. Their count alone does not
  establish instance count. The owner approved a study comparing the detector
  with a segmentation model that uses the original masks.
- **Reproduction:** Setup and commands are in `README.md`. Raw and processed
  data are Git-ignored. Record the package versions, configuration and hashes of
  any further artifacts used for reported experiments.
- **Training layout:** `prepare_yolo.py` constructs `donnees/severstal/` with
  original images, RLE annotations, separate binary masks for each class,
  the split manifest, derived image/label pairs and
  configuration files for full, 5-, 10- and 15-image rare-class training. It
  uses the same 1,252 validation and 1,266 test images for every condition.
  Image/label path checks passed for all lists, and the rare-class counts are
  5, 10 and 15. The full training list has 10,050 images; the three restricted
  lists contain 9,859, 9,864 and 9,869. Component boxes are a derived convention and mask-based evaluation
  preserves the original pixel-level ground truth.
- **Runtime dataset verification:** All 12,568 prepared image/label pairs and
  dataset lists were checked against the manifest. All 7,095 generated nonempty
  binary masks were checked pixel-for-pixel against the source RLE. The copied
  annotation CSV matches the recorded SHA-256. This verifies data preparation;
  it does not establish model quality.
- **Detector data-loader smoke test:** On the local CPU environment, PyTorch
  2.15.0+cpu and Ultralytics 8.4.216 scanned the full 10,050-image training
  list, reporting 4,721 backgrounds and zero corrupt entries. A batch of four
  training images loaded as `(4, 3, 640, 640)`. A separate sample loaded the
  15 rare-class training images and one background, with zero corrupt entries.
  No training or test evaluation was run. The effect of resizing 1,600 × 256
  images on small defects remains to be measured.
- **Segmentation data smoke test:** A temporary PyTorch dataset loaded four
  training examples, including a background, as images with shape
  `(4, 3, 256, 1600)` and four independent binary mask channels with shape
  `(4, 4, 256, 1600)`. Missing masks were treated as zero. This checks the
  files and channel convention; the actual segmentation loader and transforms
  have not been implemented.

## Remaining evidence

Record the baseline and detector configurations, training budgets and seeds,
validation metrics, rare-defect curve, threshold and cost assumptions, queue
capacity and simulated outcomes, CPU timing boundaries and hardware, final test
results, limitations, monitoring plan and deployment recommendation as those
activities occur. Keep validation decisions separate from final test results.

## Evaluation assumptions

The owner selected a hypothetical 10:1 missed-defect to false-alert cost and
an operator rate of 10–15 images per minute at 75–85% utilization. The baseline
simulation uses 12 images per minute, five seconds per image and 80% utilization
(9.6 effective per minute, conservatively rounded to nine per 60-frame window).
The cost-derived 0.091 threshold assumes a calibrated binary posterior and the
same decision unit as the costs. Box score thresholds are selected on validation.
The initial protocol and its scope are in `projet/evaluation_protocol.md`.
Literature, mathematical assumptions, model proposals and outstanding choices
are documented in `projet/study_decisions.md`.
The threshold sweep processes detections in descending score order and was
checked against independent per-threshold matching on artificial examples,
including 100 seeded synthetic cases with tied scores. Seven evaluator fixture
tests passed. Mask
overlap is accumulated per image to avoid retaining the whole dataset in memory.
These are fixture checks.

## Classical pilot

The owner selected Severstal as the sole study dataset. The assignment names
NEU-DET and its shared split, so instructor acceptance of this choice remains
unverified.

On the `model/classical` branch, an exploratory Severstal HOG/linear-SVM pilot
used 500 train and 30 validation images, seed 0. The initial random validation
sample had no instances of classes 1 or 2. The selected grid configuration was
C=1, contrast cutoff 25 and minimum area 60. Candidate recall was 6/27 at IoU
0.5. At the threshold selected on this same validation sample, TP=1, FP=9 and
FN=26, yielding 8.97 cost units per image under the hypothetical 10:1 costs.
This is preliminary and cannot establish generalization or rare-class recall.
No held-out test data was used. The pilot code was removed at the owner's
request, while its recorded local results and selected model remain in the cache.

Simulated validation lighting, noise, rotation and blur were assessed on the
same 30 images and selected threshold. Their ranges are illustrative, without
measurements from a factory line. The full-image augmentation helper transforms
boxes and masks but has not been connected to detector training. See
`../projet/mlops/README.md` for MLflow commands. MLflow stores experiment
metadata and artifacts locally in ignored `.cache/mlflow/`. The pilot import
is marked retrospective and its exact training-code snapshot is unavailable.
Future training scripts must explicitly log their experiment evidence.

The Severstal loader now reads the fixed manifest, class masks and derived boxes,
with deterministic train-only augmentation. A smoke check loaded representative
training and validation examples for each of the four classes and clean images.
This does not validate every dataset entry. Image-level binary metrics, detection
AP summaries and a callable-based inference timing helper are available for
experiments.

### Initial classical comparisons

On a seeded 500-training-image, 200-validation-image pilot, HOG with a linear
SVM achieved image-level AP 0.817 and ROC AUC 0.759. Spatial intensity and
gradient features with a random forest achieved AP 0.895 and ROC AUC 0.856.
At image-level thresholds chosen on that same validation sample using the
hypothetical 10:1 cost, the SVM produced 109 TP, 89 FP, 0 FN and 2 TN. The
forest produced 108 TP, 67 FP, 1 FN and 24 TN. These figures are exploratory,
use a relatively small sample and the thresholds are chosen on the same data.
The scores and sampled IDs are preserved in the local MLflow artifacts.

A separate region-proposal detector pilot used 500 training and 100 validation
images with three contrast/area configurations: (25,60), (15,20), (35,20).
At IoU 0.5, proposal recall was respectively 36/142, 20/142 and 25/142. The
corresponding AP50 values were approximately 0.0023, 0.0010 and 0.0004. These
figures are poor, based on a small sample, and do not establish deployment
performance. Full predictions, data IDs and settings are retained in MLflow.
Concurrent jobs ran during these pilots, making timing unsuitable for comparing
models. No held-out test data was used.

### Full-split image-level comparisons

The original full-data LinearSVC run was stopped after more than ten minutes
in fitting and marked killed in MLflow. A new run used a stochastic linear SVM
with hinge loss, balanced class weights, alpha 0.0001 and averaged SGD. Both
methods used all 10,050 training and 1,252 validation images, seed 0, original
images without augmentation and the same sampled ordering. The test split was
not opened. Scores, IDs, selected thresholds, settings and fitted models are in
local MLflow. Thresholds and metrics were selected and reported on the same
validation split, so the operating-point metrics are optimistically selected.

| Method | AP | ROC AUC | TP | FP | FN | TN | Cost/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| HOG, stochastic linear SVM | 0.828 | 0.808 | 662 | 588 | 0 | 2 | 0.470 |
| Spatial texture, random forest | 0.960 | 0.955 | 652 | 216 | 10 | 374 | 0.252 |

The cost is the hypothetical image-level 10:1 miss-to-false-alert scenario.
Both methods alerted on all 23 validation images containing class 2. This is
image alert recall, not class identification recall, and such images can contain
multiple classes. The source data has no production-run grouping, so the split
does not establish performance on independent factory runs. Data prevalence can
differ from deployment prevalence.

The 50-image decoded-image-to-score timing sample had median/p95 of 5.7/6.2 ms
for the HOG model and 40.5/56.9 ms for the forest. It excludes disk access and
is not a controlled release benchmark. Other processes were running during the
experiments. Fit times were 34.6 seconds and 2.3 seconds respectively, excluding
feature extraction and input loading. CPU memory, thread settings and full
end-to-end deployment boundaries require controlled measurement later.

## Rubric and remaining evidence

### Binary baseline tuning experiment

The fixed validation split was divided into 626 tuning and 626 comparison
images by a seeded, binary-stratified split. All four configurations trained on
the same 10,050 training images and the same original spatial texture features.
Thresholds were selected on tuning using the 10:1 image-level cost. The
comparison results below use those fixed thresholds.

| Model | Tuning cost/image | Comparison AP | Comparison TP | FP | FN | TN | Comparison cost/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random forest, 150 trees, leaf 2 | 0.252 | 0.963 | 325 | 103 | 6 | 192 | 0.260 |
| Random forest, 300 trees, leaf 1 | 0.224 | 0.964 | 324 | 86 | 7 | 209 | 0.249 |
| Extra Trees, 300 trees, leaf 1 | 0.184 | 0.970 | 326 | 81 | 5 | 214 | 0.209 |
| Histogram gradient boosting | 0.192 | 0.972 | 326 | 94 | 5 | 201 | 0.230 |

The Extra Trees configuration had the lowest tuning cost and 22 fewer false
alerts on the comparison portion than the 150-tree reference, with one fewer
miss. Both alerted on all 11 comparison images containing class 2, which is an
image-level outcome and does not establish class-identification performance.
The model family and thresholds were explored on these validation portions,
so repeated use limits their independence. The test split remains sealed. See
`../projet/training/README.md` and MLflow for configurations and image IDs.

The contrast-region proposal grid was evaluated on 100 tuning images. The
original detector setting matched 40 of 136 annotated boxes at IoU 0.5; the
best studied setting matched 46, with mean proposals increasing from 27.55 to
67.55 per image. End-to-end evaluation on the same images showed AP50 0.00261
for the original and 0.00156 for the altered proposals. Both cost-selected
thresholds produced zero true positives. These remain exploratory results.

On the same 100 tuning images and the improved proposal configuration, a
stochastic SVM trained on reference crops and random background achieved
AP50 0.00207 and zero TP at its tuned threshold. Adding matched proposals and
hard background proposals during training increased tuning AP50 to 0.01381,
with 8 TP, 56 FP and 128 FN at its validation-selected threshold. The altered
sampling also changes the random background draws. A LinearSVC hard-negative
fit was stopped after several minutes and marked killed in MLflow.

The hard-negative SGD model and its fixed tuning threshold were re-created and
evaluated on the separate 626-image validation comparison portion. Proposal
recall was 232/977 (0.237), AP50 was 0.00625, and the fixed threshold gave
16 TP, 306 FP and 961 FN. Class 2 had zero matched detections among 15 reference
boxes. The proposal and training strategy therefore remains inadequate. The
comparison portion has now been examined and remains validation data. Only 500
training images were used in this detector experiment; the test is unopened.

The professor requires a carefully tuned non-deep-learning reference and fair
comparisons on the same data and task, validation/test separation for the full
system and chosen extension, and exactly one final test measurement. Choices,
cost assumptions, operating thresholds and metrics must follow the factory need.
Production evidence must include documented CPU throughput, paired optimization
speed and quality changes, and concrete monitoring thresholds. The final analysis
must quantify failures and support its recommendation with measurements. The
repository must replay as provided in under 15 minutes, with documented code and
regular history. Test tuning, unexecuted numbers and failed replay are penalized.

Current binary methods are initial fixed configurations, and the localization
pilot is preliminary. Image-level and detection methods produce different
outputs and need task-matched comparisons on identical validation IDs. The
test has not been opened. Hyperparameter tuning, the rare-defect extension's
separate protocol, controlled CPU benchmarking, optimization trade-offs,
monitoring thresholds, quantified failure analysis and timed clean replay remain
to be completed. Instructor acceptance of the Severstal-only choice is unverified.

## AI assistance

Codex assisted with project planning, dataset research, repository and data
preparation, inspection scripts, classical pilot, MLflow integration, checks and documentation. Add further actual
uses as they occur so the owner can write the required final disclosure.
