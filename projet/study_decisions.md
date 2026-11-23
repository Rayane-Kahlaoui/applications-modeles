# Study decisions and supporting evidence

This working record supports the owner's final report. It distinguishes the
owner's approved decisions, proposed methods and evidence still to be collected.
No model performance or factory economics have been measured yet.

## Dataset and task

- **Approved:** Use the Severstal steel dataset, the recorded image-level
  train/validation/test split and class 2 as the rare class. Retain 5, 10 and 15
  class-2 training images in nested subsets. Use the full-data detector as a
  reference. Preserve the same target-free training images across conditions.
- **Approved:** Compare a classical reference, a learned detector and a
  segmentation model. Preserve original masks for mask evaluation and use
  derived connected-component boxes for the initial detector comparison.
- **Limitation:** A disconnected mask region is a derived component, not a
  verified physical defect. The numeric class names in the public annotations
  have not been verified as engineering defect names. The data lacks a
  documented production-run grouping, and its prevalence need not represent
  the factory. See `../DataPrep/report_evidence.md` for the audit and counts.

## Cost-sensitive decision policy

- **Approved scenario:** Assign a hypothetical cost of 10 to a missed reference
  box and 1 to a spurious predicted box. This encodes the study's assumption
  that escaping a defect is more costly than investigating a false alert. It
  is not a measured factory cost and applies to derived boxes, whose number can
  depend on annotation conversion. Report TP, FP, FN and cost per image, as well
  as image-level outcomes. An image-level economic model needs separately
  specified costs; an operator reviews an image, potentially containing several
  boxes, once.
- **Probability threshold:** For a binary action with zero equal costs for
  correct decisions, false-positive cost `C_FP`, false-negative cost `C_FN`
  and calibrated deployment posterior `p`, alerting costs `C_FP * (1-p)` in
  expectation and not alerting costs `C_FN * p`. Alert when
  `p >= C_FP / (C_FP + C_FN)`. Under 10:1, this is 1/11, about 0.091. The
  probability and costs must refer to the same event and action. Other cost
  matrices require their full decision calculation. Elkan derives the general
  cost-sensitive rule and discusses probability adjustment and thresholding.
  [Elkan, The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
- **Application to this study:** Detector scores are not established as
  calibrated deployment probabilities. Select thresholds by the predeclared
  validation sweep and record the score floor. Compare cost ratios 5:1, 10:1
  and 20:1 as hypothetical sensitivity scenarios. Freeze the selected policy
  before the final test. The study of network intrusion detection uses those
  ratios and finds its own model's optimal validation thresholds differ; this
  supports sensitivity analysis in an asymmetric-cost setting, not a universal
  steel-inspection ratio. [Cha et al., Electronics, 2026](https://www.mdpi.com/2079-9292/15/7/1542)
- **Another operational example:** A freight-cancellation study uses a
  cost-implied probability threshold around 0.091 and emphasizes checking how
  predictions translate into actions. Its domain and costs differ from steel
  inspection; it does not measure this factory's cost ratio. [Cost-Sensitive
  Architecture Auditing for LLM-Based Tabular Prediction](https://www.sciencedirect.com/science/article/pii/S2666827026001623)
- **Prevalence and capacity:** Report sensitivity to deployment prevalence and
  false-alert workload. Under prior shift, a posterior calibrated to the
  sampled dataset may need recalibration for deployment. If the posterior is
  already calibrated to the deployment population and the cost matrix is
  unchanged, the theoretical probability cutoff remains 1/11; prevalence
  changes the posterior and expected alert volume, not that formula. Increasing
  `C_FN/C_FP` lowers the theoretical cutoff. A higher score threshold may be
  selected under a different cost matrix, uncalibrated scores or an explicit
  capacity constraint. Weighting scenarios require stated prevalence and
  transport assumptions, and must not be presented as measured factory rates.
- **Manufacturing context:** A single-line automotive study illustrates that
  defect prevalence, false-alarm volume, temporal changes and capacity can
  affect operational cost. Its findings are specific to its case and are not
  evidence that 10:1 is a general manufacturing standard. [Tkocz and Sikora,
  Stage-Wise Threshold Calibration, 2026](https://www.mdpi.com/2504-4990/8/10/324)
- **Unverified source:** The suggested semiconductor cost-matrix article at
  https://clouatre.ca/posts/accuracy-cost-matrix-threshold could not be accessed
  during this review. The proposed 14:1 figure and claimed savings are not
  verified and should not be cited as results without checking the original.

## Operator review

- **Approved range:** 10–15 images per minute at 75–85% utilization. These are
  owner-provided planning assumptions, not measurements of an operator.
- **Baseline choice:** 12 images/minute, five seconds/image and 80% utilization
  give 9.6 effective reviews/minute. The current discrete simulation accepts
  nine per 60-frame window at one frame/second, rounding down. It handles queued
  images in input order, resets capacity at each window and leaves overflow on
  the automatic policy. This is not a continuous queue model.
- **Simulation boundary:** Annotation-perfect review is an optimistic bound
  only for accepted images. Count images with no retained boxes as bypassing
  the queue and report their misses. Review-band selection, actual operator
  accuracy and review time require validation before deployment.

## Models and decisions before training

- **Classical reference — proposed:** Candidate regions from local contrast,
  morphology and connected components, then HOG features and a linear SVM
  trained on defect and background crops. Tune a small declared grid on
  validation, record candidate recall and preserve fitted artifacts. This
  provides a non-deep-learning detector with the same box output schema.
  Proposal settings, crop sampling, background definition, model dependencies
  and grid remain to be specified and reviewed before fitting.
- **Detector — proposed:** A small pretrained nano YOLO variant with generic
  initialization. Select the exact version, license, pretrained weight source,
  input resolution and device before training. A short pilot will measure CPU
  time and memory before the owner approves the full training budget. Compare
  full-image resizing and any tiling strategy on training/validation, including
  their end-to-end cost. Group all tiles by source image. The held-out test is
  reserved.
- **Segmentation — approved comparison, architecture undecided:** Select a
  compact architecture and initialization after defining the multi-label mask
  loader, binary channel handling and image-level alert score. Use identical
  image IDs and measure per-class mask overlap, image-level outcomes and CPU
  time. The exact architecture, loss, resolution and training budget remain
  decisions. Do not imply mask IoU and box AP are directly interchangeable.
- **Rare learning curve:** Reinitialize rare-class detector fits from generic
  weights, not a checkpoint trained with excluded rare-class images. The
  segmentation rare-class protocol and budget remain to be specified if it is
  included in the same curve. A single subset ordering is exploratory.
- **Evaluation and release:** Confirm the segmentation image-score adapter,
  score floor, cost units, operating policy, capacity, pretrained artifact
  license, training budget, reproducible environment and exact submission
  deadline. Freeze all variants, thresholds, code and artifacts after
  validation before opening the test once. The instructor invitation remains
  the owner's task.

## Evidence still required

Record dataset and preprocessing hashes, training configurations and elapsed
time, validation results, score calibration, sensitivity analyses, queue
workload, CPU latency, failure examples, final test evidence, monitoring plan
and actual AI assistance as the project progresses. The owner writes the final
report in their own words.
