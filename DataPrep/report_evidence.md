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

The owner selected Severstal as the sole study dataset and clarified that any
dataset may be used. The written assignment names NEU-DET and its shared split.
This clarification is recorded as the owner's information.

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

## Image-level CNN comparison

### Scratch CNN and synthetic condition checks

The user requested a custom model from scratch without fine-tuning pretrained
backbones. A four-block CNN with average and maximum pooling, AdamW and training
augmentation was trained on 9,045 of the 10,050 training images. The remaining
1,005 training images selected the checkpoint by average precision. Eight epochs
were run, and epoch 8 had the best training-holdout AP of 0.9474. Loading and
caching took 75.0 seconds, and training with holdout evaluation took 768.5
seconds on CPU. The same training pool and same validation IDs were verified
against the forest run. The CNN's gradient-update set is smaller than the forest's
full training set due to its training-only checkpoint holdout.

At the threshold selected on 626 tuning images, the same 626 comparison images
gave 328 TP, 133 FP, 3 FN and 162 TN, AP 0.957 and cost/image 0.260 under the
hypothetical image-level 10:1 cost. All 11 comparison images containing class 2
were flagged, which is image-level recall, not defect-type identification. The
earlier Random Forest had 324 TP, 86 FP and 7 FN at cost/image 0.249. The
comparison data has been examined repeatedly and the final test remains sealed.

For the CNN, fixed-threshold stress tests on the same comparison images gave:

| Condition | TP | FP | FN | TN | AP | Cost/image |
|---|---:|---:|---:|---:|---:|---:|
| Original | 328 | 133 | 3 | 162 | 0.957 | 0.260 |
| Darker, multiplier 0.75 | 327 | 139 | 4 | 156 | 0.953 | 0.286 |
| Brighter, multiplier 1.25 | 325 | 138 | 6 | 157 | 0.952 | 0.316 |
| Gaussian noise, std 8/255 | 320 | 119 | 11 | 176 | 0.920 | 0.366 |
| Gaussian blur, sigma 1 | 330 | 196 | 1 | 99 | 0.932 | 0.329 |

These are illustrative transforms applied after resize, not measurements of an
actual factory line. Blur caused 63 additional false alerts relative to the
original images, and the noise scenario increased misses by eight. A four-thread
CPU timing pass on the same 100 decoded comparison images as prior models gave
median 3.3 ms and p95 4.0 ms through preprocessing and scoring, excluding disk.
It is exploratory rather than a complete deployment benchmark. Caching and
training together took about 14 minutes, before other replay work. A complete
clean-clone replay has not been timed; saved weights can avoid retraining.

### Training-only Random Forest search

Eight Random Forest hyperparameter combinations were evaluated using seeded,
binary-stratified three-fold cross-validation on the training images only.
The selection metric was mean average precision. The selected configuration
used 150 trees, leaf size 1, `sqrt` feature sampling, unlimited depth and
`balanced_subsample` class weights. Its mean training-fold AP was 0.9481.
After refitting on all 10,050 training images, the image-cost threshold was
chosen on the same 626 validation tuning images used previously, then applied
to the same 626 comparison images. It achieved comparison AP 0.9638, 326 TP,
124 FP, 5 FN and 171 TN, yielding 0.278 cost/image under the hypothetical 10:1
image costs. The earlier 300-tree forest had 324 TP, 86 FP, 7 FN and 209 TN,
with 0.249 cost/image. The training-fold search did not improve false alerts
on this repeatedly examined comparison subset. The test remains unopened.

### Threshold trade-off from saved validation scores

The highest score threshold achieving each target image-level recall was chosen
on the 331 positive tuning images and applied to the separate comparison portion
of 331 positive and 295 negative images. For the 300-tree Random Forest, a 99%
tuning-recall target gave 326 TP, 108 FP, 5 FN and 187 TN on comparison. A 98%
target gave 323 TP, 77 FP, 8 FN and 218 TN. A 95% target gave 306 TP, 48 FP,
25 FN and 247 TN. These targets are achieved on tuning; observed recall differs
on comparison. At the original 10:1 cost-selected threshold the same forest had
324 TP, 86 FP, 7 FN and 209 TN. For the 95% target, eliminating 38 false alerts
while adding 18 misses increases hypothetical 10:1 cost from 156 to 298 on
these 626 images. This analysis reused the comparison portion and is exploratory.
Results are image-level and the target does not guarantee rare-class recall.
The held-out test remains unopened.

The owner selected a Random Forest reference and two pretrained CNNs. The
professor's Day 1 vision/transfer-learning course names EfficientNet-B0 and
ResNet-18 and compares pretrained feature extraction. MobileNetV3-Small was
chosen as the smaller second CNN. This is a binary image-alert experiment.

The forest is the earlier 300-tree, leaf-size-1 configuration. Both CNNs use
frozen ImageNet-1k weights and a standardized class-balanced logistic regression
head fitted on the training data. Grayscale images are resized to 256 by 64,
repeated into three channels and ImageNet-normalized. This uses the full image
and differs from the original ImageNet preprocessing. All three used exactly
the same 10,050 training IDs, 626 validation tuning IDs and 626 validation
comparison IDs, verified from their recorded results. Thresholds minimize the
hypothetical 10:1 image cost on tuning. Comparison metrics use fixed thresholds.

| Model | Tuning cost | Comparison AP | ROC AUC | TP | FP | FN | TN | Comparison cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.224 | 0.964 | 0.960 | 324 | 86 | 7 | 209 | 0.249 |
| MobileNetV3-Small | 0.220 | 0.954 | 0.951 | 327 | 129 | 4 | 166 | 0.270 |
| EfficientNet-B0 | 0.316 | 0.938 | 0.942 | 329 | 164 | 2 | 131 | 0.294 |

Each model alerted on all 11 comparison images containing class 2. This is
image-level alert recall, including multi-class images, not type recognition.
The comparison portion has been examined repeatedly. A single seed and the
unavailable production-run grouping limit conclusions. Test remains unopened.

An exploratory CPU timing pass used the same first 100 comparison images, five
warm-up predictions and four configured threads. Median/p95 decoded-grayscale
image-to-score times were 49.8/60.4 ms for the forest, 7.4/9.2 ms for MobileNet,
and 20.3/24.5 ms for EfficientNet on an AMD Ryzen 7 4800H. Disk loading is
excluded. These measurements are not an isolated end-to-end factory benchmark.
Training image loading and frozen feature extraction took 87.6 seconds for
MobileNet and 157.3 seconds for EfficientNet; head fitting took 0.73 and 1.53
seconds respectively. The forest's original full-split run used the same data
but recorded feature extraction and fit times separately in MLflow. CNN weights,
hashes, source, fitted heads, IDs and scores are retained as local artifacts.

### Expanded image-level comparison for the report

The following values are calculated from the recorded comparison confusion
matrices. The common comparison subset contains 626 images, with 331 annotated
defective and 295 without annotated defects. Each threshold was selected on the
separate 626-image tuning subset using a false-negative cost of 10 and a
false-positive cost of 1. Cost per image is `(10 * FN + FP) / 626`.

| Model | TP | FP | FN | TN | Recall | Precision | Specificity | False-alert rate | F1 | Accuracy | AP | Cost/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees, 300 trees, leaf 1 | 326 | 81 | 5 | 214 | 98.5% | 80.1% | 72.5% | 27.5% | 88.3% | 86.3% | 97.0% | 0.209 |
| Random Forest, 300 trees, leaf 1 | 324 | 86 | 7 | 209 | 97.9% | 79.0% | 70.8% | 29.2% | 87.4% | 85.1% | 96.4% | 0.249 |
| MobileNetV3-Small, frozen | 327 | 129 | 4 | 166 | 98.8% | 71.7% | 56.3% | 43.7% | 83.1% | 78.8% | 95.4% | 0.270 |
| EfficientNet-B0, frozen | 329 | 164 | 2 | 131 | 99.4% | 66.7% | 44.4% | 55.6% | 79.9% | 73.5% | 93.8% | 0.294 |

Recall is TP/(TP+FN), precision is TP/(TP+FP), specificity is TN/(TN+FP),
false-alert rate is FP/(FP+TN), F1 is 2TP/(2TP+FP+FN), and accuracy is
(TP+TN)/626. AP summarizes the precision-recall curve across thresholds. The
other confusion-matrix metrics use the fixed tuning-selected threshold. The
observed precision and workload depend on prevalence. These are binary image
alerts and do not measure localization or class identification.

Across all recorded candidates, the four lowest comparison costs were Extra
Trees at 0.209, histogram gradient boosting at 0.230 (326 TP, 94 FP, 5 FN,
201 TN, AP 0.972), Random Forest at 0.249, and histogram gradient boosting
with multiscale features at 0.252 (325 TP, 98 FP, 6 FN, 197 TN, AP 0.974).
The scratch CNN followed with 328 TP, 133 FP, 3 FN, 162 TN, AP 0.957 and
cost/image 0.260. These are candidate-run rankings, including two feature
variants of gradient boosting. The binary experiment outputs and predictions
are retained in the local ignored `.cache/binary-tuning-v1/`,
`.cache/multiscale-binary-v1/`, `.cache/mobilenet-binary-v1/`,
`.cache/efficientnet-binary-v1/` and `.cache/custom-cnn-v1/` directories.

The same comparison subset has informed multiple experiments, so its ranking
is exploratory. Final test data remains unopened. CPU timings above exclude
disk loading, and a comparable Extra Trees timing is not recorded here.

### References and candidates selected for fine-tuning

Keep Extra Trees as the primary non-deep-learning image-alert reference because
it achieved the lowest observed comparison cost, 0.209 per image, with 81 false
alerts and five misses. Retain the previously selected Random Forest as an
additional classical reference, with 86 false alerts, seven misses and cost
0.249. Both remain subject to the limitations of repeated validation use.

Fine-tune MobileNetV3-Small first, motivated by its 327/331 comparison recall
and 7.4 ms median CPU decoded-image-to-score time, then investigate
EfficientNet-B0, which missed two comparison images and had a 20.3 ms median.
Their frozen-feature runs remain references for measuring fine-tuning gains
and losses. MobileNet and EfficientNet produced 129 and 164 false alerts,
respectively, so fine-tuning must be measured rather than assumed to help.

Histogram gradient boosting and its multiscale variant remain documented
exploratory comparisons. Their costs of 0.230 and 0.252 exceeded Extra Trees,
although the multiscale variant had AP 0.974. The scratch CNN remains a
documented exploratory comparison, with three misses, 133 false alerts and
cost 0.260. The cross-validated forest and multiscale Extra Trees variants also
remain documented, at costs 0.278 and 0.264 respectively. These selections are
for binary image alerts and do not establish localization, type recognition,
rare-class performance or a final production model. The test remains unopened.

### Larger frozen CNN comparisons

On 2026-11-23, MobileNetV3-Large and EfficientNet-B2 were compared using the
same full-image 256-by-64 preprocessing, ImageNet normalization, frozen
backbones and standardized, class-balanced logistic heads as the earlier CNNs.
All five runs in the table used identical ordered training (10,050), validation
tuning (626) and comparison (626) IDs, verified from their saved results.
Thresholds minimized the assumed 10:1 image-level cost on tuning. The
comparison portion contains 331 defective and 295 defect-free images.

| Model | Tuning cost/image | Comparison TP | FP | FN | TN | AP | Precision | Recall | F1 | Comparison cost/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra Trees | 0.184 | 326 | 81 | 5 | 214 | 0.970 | 0.801 | 0.985 | 0.883 | 0.209 |
| MobileNetV3-Small | 0.220 | 327 | 129 | 4 | 166 | 0.954 | 0.717 | 0.988 | 0.831 | 0.270 |
| MobileNetV3-Large | 0.268 | 330 | 162 | 1 | 133 | 0.946 | 0.671 | 0.997 | 0.802 | 0.275 |
| EfficientNet-B0 | 0.316 | 329 | 164 | 2 | 131 | 0.938 | 0.667 | 0.994 | 0.799 | 0.294 |
| EfficientNet-B2 | 0.297 | 330 | 160 | 1 | 135 | 0.946 | 0.673 | 0.997 | 0.804 | 0.272 |

Both new CNNs alerted on all 11 comparison images containing class 2. This is
image-level alert recall, including mixed-class images, and does not measure
defect type identification. MobileNetV3-Small remains the lightweight CNN
reference because its selected tuning and comparison costs are lower than the
Large variant's. EfficientNet-B2 becomes the EfficientNet frozen-feature
reference for future comparisons because its tuning and comparison costs and
AP improved over B0, with one fewer comparison miss and four fewer false
alerts. Retain B0 and MobileNetV3-Large results as additional evidence. Extra
Trees remains the primary classical reference. None of these selections is a
final test result or a claim that the same ranking will hold in production.

The new official Torchvision weights were downloaded from
`https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth` (SHA-256
`5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`) and
`https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth`
(SHA-256 `c35c147384e385a5bab5a8eabdabbe5a3df0487ee4a554108626ae474a5bf755`).
The saved heads, scores, IDs, metrics and weights are in the local MLflow runs
and ignored output directories `.cache/mobilenet-large-binary-v1/` and
`.cache/efficientnet-b2-binary-v1/`. The respective MLflow runs include the
weight files. The source snapshots record the runner used for these experiments.

On the same first 100 comparison images, with five warm-ups and four CPU
threads, decoded-image-to-score median/p95 timing was 12.9/14.9 ms for Large
and 28.2/35.9 ms for B2. Earlier Small and B0 timings were 7.4/9.2 ms and
20.3/24.5 ms respectively. Timing excludes disk access and is exploratory;
the measurements were made in separate sessions and are not end-to-end factory
benchmarks. Repeated selection on validation and unknown factory prevalence
limit interpretation. The test remains unopened.

## Rubric and remaining evidence

### Classical localization follow-up

On 2026-11-23, the region detector was extended with three interleaved contrast
proposal scales, a combined HOG/local-binary-pattern/shape feature set and a
balanced 150-tree Extra Trees crop classifier. Proposals and hard backgrounds
were sampled from training images. The validation tuning study used the same
first 100 tuning images and 136 reference boxes as the earlier proposal study.
Using 500 training images, the new proposals covered 56/136 boxes at IoU 0.5,
versus 46/136 for the best earlier proposal grid. The combined 500-image
texture/SGD run reached AP50 0.01415 and the Extra Trees run 0.01446. The SGD
fit reached its iteration limit. Multiple settings changed together, so their
individual contributions are not established.

Using 2,000 training images with the same proposal and Extra Trees settings gave
13 TP, 69 FP, 123 FN and AP50 0.01597 on those 100 tuning images. Training on
all 10,050 images gave 21 TP, 118 FP, 115 FN and AP50 0.01840. Both thresholds
minimized the 10:1 per-box error cost on these tuning images. The full-data fit
required 540.0 seconds for training feature extraction and 79.8 seconds for
classifier fitting. One attempt to score every class per proposal with the
2,000-image model gave the same tuning-selected counts, so the simpler scoring
path was retained. Its source snapshot and results remain in MLflow.

The fitted models and thresholds were then evaluated without refitting on the
same 626 comparison images used by the earlier classical and YOLO pilots.
Comparison results were:

| Candidate | Training images | TP | FP | FN | AP50 | Box cost/image |
|---|---:|---:|---:|---:|---:|---:|
| Earlier classical pilot | 500 | 16 | 306 | 961 | 0.00625 | 15.84 |
| Multiscale texture Extra Trees | 2,000 | 86 | 420 | 891 | 0.01789 | 14.90 |
| Multiscale texture Extra Trees | 10,050 | 127 | 730 | 850 | 0.02228 | 14.74 |

The comparison sample contains 977 component-derived reference boxes. The
full-data model's 127 matches were all class 3. It matched none of the 159
class-1, 15 class-2 or 87 class-4 boxes at the selected threshold. Its
proposal recall was 323/977, before class scoring, and the comparison median/p95
decoded-image inference timing was 0.222/0.252 seconds with no warm-up. This
timing is exploratory and excludes disk. A YOLO pilot trained on only 500 images
previously reached AP50 0.02754 on these same images, with its own training
regime. The different training sizes and settings limit interpretation.

These runs are retained in local MLflow and `.cache/region-multiscale-*` along
with fitted models, predictions, IDs and settings. In the two saved-model
comparison runs, the `hard_negatives` parameter and configuration recorded the
evaluation command's default `false`, even though the source training runs used
hard negatives. The `source_run_id` links to the correct training records. The
runner was corrected after these comparisons to record the source setting.
Repeated validation use limits independence, the test remains unopened, and the
rare-class result and false-alert burden show that this is still a weak
localization baseline. A clean replay and controlled end-to-end CPU benchmark
remain outstanding.

### Cycle 4 baseline selection and replay

The selected class-aware classical reference at this point is the full-training
multiscale texture/Extra Trees detector. Its tuning run is
`a371f915bc8c487d86b01c34a564b731` and its fixed-threshold comparison run is
`3c9bd18408d34c3f8333fbea26f84534`. It achieved AP50 0.02228, 127 TP, 730 FP,
850 FN and cost/image 14.744 on 626 comparison images. All 127 matched boxes
were class 3. This weak result is the measured classical reference and a fair
future detector comparison must use the same images, reference boxes, matching
rules and task-level metrics. AP50 and box-level cost differ from image-level
classification AP and cost.

As a separate image-alert reference, Extra Trees produced 326 TP, 81 FP, five
FN, 214 TN, AP 0.97009 and cost/image 0.20927 on the same 626 images. The
frozen-feature MobileNetV3-Small and EfficientNet-B2 runs remain CNN references.
These binary runs cannot provide defect type or location.

On 2026-11-23, the selected localization model was reloaded from
`.cache/region-multiscale-trees-full-v1/model.joblib`. The first five comparison
images, the same processing settings and the threshold -0.09999999999999998
reproduced the originally saved ground truth and prediction lists exactly.
Replay run ID is `aa32450266224b1c8de83cc9291c6269`. This is a local five-image
check, not a clean-clone replay. The fitted model is 595,447,866 bytes and the
binary Extra Trees model is 100,731,513 bytes. Both remain in the ignored local
cache and MLflow artifacts, and require an appropriate distribution method for
the instructor. The SVM regularization grid remains untested. Repeated
validation examination and unverified production-run independence limit
generalization. The test split remains unopened.

### Linear SVM regularization check

The stochastic hinge-loss SVM was compared with three alpha values on the same
500 training images, multiscale proposals, texture features and 100 tuning
images. Alpha 0.00001 yielded TP 7, FP 68, FN 129, AP50 0.01549 and cost per
image 13.58. Alpha 0.0001 yielded TP 4, FP 37, FN 132, AP50 0.01415 and cost
13.57. Alpha 0.001 yielded TP 3, FP 19, FN 133, AP50 0.01455 and cost 13.49.
The 0.0001 and 0.001 fits reported reaching the iteration limit. Thresholds
were each selected on these same tuning images, and no final-test data was
used. This small grid improves documentation of classical tuning but does not
establish a global optimum or a class-balanced detector. The chosen full-data
Extra Trees model had tuning cost 12.68 under the same 10:1 per-box assumption,
with a larger training budget. Experiment configurations and predictions are
retained locally in MLflow.

Selected model copies, weight hashes and validation predictions have been
prepared under `projet/artifacts/baselines/` for Git LFS distribution. These
support inspection of selected evidence, while the complete MLflow store still
requires a portable export. A clean-clone replay has not yet been verified.

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

### Multiscale image-level features

An additional feature representation preserves the source aspect ratio at
512 by 82 and uses local mean and standard deviation of intensity, gradient and
local contrast at two spatial scales, plus an intensity histogram. It was
evaluated on the same 10,050 training and 626/626 validation partitions.
With Extra Trees, tuning cost was 0.212 and comparison cost was 0.264 per image,
with 326 TP, 115 FP, 5 FN and 180 TN. Comparison AP was 0.964. The earlier
feature representation gave tuning cost 0.184 and comparison cost 0.209, with
326 TP, 81 FP, 5 FN and 214 TN. Multiscale Extra Trees alerted on 10 of 11
comparison images containing class 2, compared with 11 of 11 previously.

The histogram-gradient-boosting variant with the new features had tuning cost
0.219 and comparison cost 0.252, with 325 TP, 98 FP, 6 FN and 197 TN. It had
comparison AP 0.974. Neither multiscale variant improved on the earlier Extra
Trees operating cost on these partitions. These are repeated validation
experiments, and only the test split remains sealed.

### Pretrained detection pilot

YOLO11n initialized from the official Ultralytics pretrained weights was tested
on CPU using exactly the same 500 training and 100 validation tuning images as
the hard-negative classical region detector. Initial weights SHA-256 was
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
The image size was 320 and Ultralytics training defaults, including its own
augmentation settings, were used. A one-epoch run gave AP50 0.00740. A separate
five-epoch run gave tuning AP50 0.03455, 16 TP, 57 FP and 120 FN at the
threshold selected on those same images. Training took 249 seconds on the local
CPU, including framework training and validation. The short pilots are not
comparable controlled training-time benchmarks.

With the five-epoch checkpoint and tuning-selected threshold fixed, evaluation
on the same 626 comparison images as the classical detector gave AP50 0.02754,
108 TP, 380 FP and 869 FN. It detected zero of the 15 class-2 reference boxes.
The classical hard-negative detector had AP50 0.00625, 16 TP, 306 FP and 961 FN
on those images. Both used 500 training images, but their preprocessing and
augmentation differ. The checkpoint was selected using the tuning images. These
are validation comparisons and have been inspected repeatedly, and the models
remain weak at localization. The test split remains unopened. Ultralytics lists
YOLO11 under AGPL-3.0 and enterprise licenses; assess licensing before use beyond
this study. Official model documentation: https://docs.ultralytics.com/models/yolo11

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
to be completed. The owner clarified that any dataset may be used.

## AI assistance

Codex assisted with project planning, dataset research, repository and data
preparation, inspection scripts, classical pilot, MLflow integration, checks and documentation. Add further actual
uses as they occur so the owner can write the required final disclosure.
