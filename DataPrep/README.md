# Severstal data preparation

Study decisions, checks and limitations are recorded in `report_evidence.md`.

Run commands from the project root in PowerShell. Create the Python environment
as described in the root README, then install the project packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sign in to Kaggle and accept the competition rules on the
[competition page](https://www.kaggle.com/competitions/severstal-steel-defect-detection).
Authenticate with `.\.venv\Scripts\kaggle.exe auth login`, then download:

```powershell
.\.venv\Scripts\kaggle.exe competitions download -c severstal-steel-defect-detection -p DataPrep/raw
```

Extract the competition archive and any nested training-image archive under
`DataPrep/raw/`, so `DataPrep/raw/train.csv` and
`DataPrep/raw/train_images/*.jpg` exist. The original competition test images
are not used in this project's labelled split.

`raw/` holds downloaded source files and `processed/` holds intermediate data.
Both are ignored by Git. The approved split manifest is tracked in `DataPrep/`.

With `train.csv` and `train_images/` directly under `raw/`, run the integrity
inspection from the project root:

```powershell
.\.venv\Scripts\python.exe DataPrep/inspect_data.py
```

It checks image decoding, dimensions, CSV rows, RLE bounds, byte-identical files
and identical decoded pixels. Related images may remain; the supplied metadata
does not identify production runs or coils.

`manifests/split.json` records an image-level split with seed 0, stratified by
each image's full set of classes. It also contains nested 5-, 10- and 15-image
training conditions for class 2. These conditions keep all images without class
2 and exclude every other image containing that class, including mixed-class
images. Keep the tracked manifest for reproducible experiments. The split script
will not overwrite an existing manifest.

Run `.\.venv\Scripts\python.exe DataPrep/masks_to_boxes.py` to generate YOLO
labels under `processed/labels/`. A class mask is split into 8-connected pixel
components, each of which receives one bounding box. Boxes use zero-based YOLO
class IDs 0–3 corresponding to dataset classes 1–4. Coordinates are normalized
to the original image dimensions. Each image has a label file, including an empty
file for images without annotated defects. Components are a reproducible box
convention and may not correspond to distinct physical defects. The conversion
will not overwrite an existing labels directory.

After generating the labels, build the Ultralytics layout:

```powershell
.\.venv\Scripts\python.exe DataPrep/prepare_yolo.py
```

This creates `../donnees/severstal/` with `images/`, `labels/`, `masks/`,
`annotations.csv`, `split.json`, image lists and `full.yaml`, `rare_5.yaml`,
`rare_10.yaml` and `rare_15.yaml`. Each configuration points to its training list
and the shared validation and test lists. `masks/class_1/` through `class_4/`
contain lossless binary PNG masks, named after the source image. An absent file
for an image/class pair denotes an empty mask. Separate files preserve overlaps
between classes. The original RLE annotations are in `annotations.csv`.

The configurations contain absolute paths for this machine. Regenerate the
layout after moving the project. The script will not overwrite an existing
layout. It uses hard links when available and copies when they are not.
Hard-linked image and label files share their contents with the source files, so
treat them as read-only. `donnees/` is ignored by Git and can be rebuilt from
the source data using these commands and the tracked manifest.

The Ultralytics 8.4.216 detector loader scanned all 10,050 training images and
loaded a batch successfully. Install its CPU dependencies from the project root
with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
The recorded environment has not yet been verified through a clean install. Use
the test list only in the approved final evaluation.

The individual class masks can be read as four binary channels, treating a
missing image/class mask as all zeros. A small PyTorch smoke test verified this
layout; the project segmentation loader and preprocessing are still to be
implemented. A 640-pixel detector smoke test resized the long images, so the
effect on small defects still needs to be evaluated on training and validation
data before selecting the production input strategy.
