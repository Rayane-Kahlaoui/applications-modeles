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
These are fixture checks; no model results are available yet.

## AI assistance

Codex assisted with project planning, dataset research, repository and data
preparation, inspection scripts, checks and documentation. Add further actual
uses as they occur so the owner can write the required final disclosure.
