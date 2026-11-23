# Evaluation protocol

## Hypothetical decision assumptions

The owner selected a baseline cost ratio of 10 units per missed defect and one
per false alert. For a binary action with a calibrated posterior probability
`p`, equal true-decision costs and only false-positive and false-negative costs,
the Bayes decision threshold is `1 / (1 + 10) ≈ 0.091`. This is an image-level
illustration when the costs and probability refer to an image-level event.
The detector's raw box scores are not assumed to be calibrated probabilities.
Select box-score thresholds using validation cost rather than applying 0.091 to
them automatically. Evaluate alternative cost and deployment-prevalence
scenarios on validation. A change in prevalence can change the posterior and
expected costs; do not apply the prior twice to an already calibrated posterior.
The dataset's sampled prevalence need not match the production line.

The costs in the current box evaluator apply to derived connected components,
not verified physical defect instances. The 10:1 ratio is a hypothetical
scenario chosen by the owner, not a measured steel-industry cost ratio. A
separate image-level cost calculation requires its own assumptions because one
image may contain multiple boxes and an operator reviews an image once. See
`study_decisions.md` for the literature and outstanding decisions.

The selected operator range is 10–15 images per minute (4–6 seconds per image)
at 75–85% utilization. The baseline scenario uses 12 images per minute, five
seconds per image, and 80% utilization: 9.6 effective images per minute. For
the discrete 60-frame simulation use nine reviews per 60-frame window, with
the first nine queued images accepted. This conservatively rounds down the
baseline. At one frame per second, report queue images per 60 frames and
simulated review time for each accepted image. Additional scenarios span the
effective range 7.5–12.75 images per minute. The window-reset policy is an
approximation and does not model a continuous backlog.

## Shared inputs

For box evaluation, an ordered mapping uses one entry per image, including
images with no defects or predictions. Each value has `ground_truth` and
`predictions` lists. A ground-truth box has `class_id` and `box`; a prediction
also has `score`. Boxes are `[x_min, y_min, x_max, y_max]` in the same image
coordinate system. IDs use one consistent mapping per experiment. Retain the
full image inventory and the prediction score floor alongside cached results.
Before evaluation, compare image IDs with the selected manifest. The evaluator
does not infer missing images from a prediction file.

## Matching and thresholds

Match each prediction in descending score order to the highest-IoU unmatched
ground-truth box of the same class in the same image at IoU ≥ 0.5. Duplicates
are false positives. A wrong class can create both a false positive and a miss.
Equal scores within an image retain input order; inputs must be deterministic.
For per-class AP, equal scores are ordered by image ID and then input order.
AP50 uses 101-point interpolated precision over retained predictions. A class
without ground truth has undefined AP and recall. Undefined precision is
reported when there are no positive predictions.

Thresholds use `score >= threshold`, including just below the smallest retained
score and just above the largest. For each threshold report TP, FP and FN and
`(miss_cost * FN + false_alert_cost * FP) / image_count`. Choose minimum cost,
then fewer misses, then the higher threshold. Results depend on the retained
score floor and on box components as the derived ground truth.
`evaluate_boxes` returns the sweep, selected threshold, class metrics,
image-alert counts, score floor and cost assumptions. The sweep processes each
retained prediction once in descending score order. AP is computed over the
retained scores, so a restrictive prediction floor can lower measured AP.

## Masks, images and review

For binary class masks, stream per-image arrays of shape `(classes, height,
width)` and aggregate pixel TP, FP and FN across images per class,
then compute IoU and Dice. When both prediction and ground truth are empty,
overlap is undefined. Preserve independent class channels. Image-level alerts
count an image as positive when any annotated class is present and alert when
any retained box reaches the automatic threshold. For segmentation, adapters
must provide a separately documented image-level score and threshold; these
must be selected on validation.

For the box-review simulation, queue an image if any box score is in
`[lower, upper)`, including images with another high-score box. Images without
retained boxes bypass the queue. Reviewed images are treated as perfectly
corrected for an optimistic bound; overflow stays automatic. Count misses on
bypassed images. The evaluator uses deterministic input order and a per-window
capacity. The chosen band, review time, workload cap, model and score floor
must be frozen before any final test.

## Checks

Image-level `binary_metrics` reports counts, precision, recall, F1, specificity,
false-positive rate, average precision and ROC AUC from continuous scores.
Undefined values are `None`; report class support and the selected threshold.
Average precision summarizes the precision-recall curve and is not trapezoidal
PR-AUC. `detection_summary` reports per-class F1, AP50 and mean AP over IoUs
0.50 through 0.95 in 0.05 steps, using the supplied predictions and 101-point
interpolation. This is not an official COCO evaluator, which has additional
area and maximum-detection conventions. Classes without ground truth are omitted
from AP means. Score floors and NMS can affect AP and must be reported.

`benchmark_inference` measures the supplied callable, after warm-up, with
per-image, mean, median, p95 and maximum wall times. To measure end-to-end CPU
latency, the caller must include preprocessing, model inference and postprocessing
in that callable and record the hardware, thread settings, sample IDs and timing
boundaries separately. Training duration must be measured around each training
run. Record metrics and timing in MLflow from the model-specific scripts.

Run `python -m unittest projet.tests.test_evaluation` from the
repository root. These are hand-checkable fixtures, not model results. Do not
open the held-out test while designing the protocol.
