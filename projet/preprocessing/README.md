# Severstal loader and augmentation

`SeverstalDataset` reads the prepared dataset and its fixed split manifest from
`donnees/severstal/`. It returns a grayscale uint8 image at source resolution,
zero-based class labels, pixel-coordinate xyxy boxes, four independent binary
class masks and the image ID. Independent masks preserve overlaps. Empty label
files represent images without annotated defects. Labels and masks are checked
for class consistency when an image is loaded.

```python
from torch.utils.data import DataLoader
from projet.preprocessing.severstal import SeverstalDataset, collate_samples

train = SeverstalDataset("train", augment_training=True, seed=0)
validation = SeverstalDataset("validation")
train.set_epoch(0)
loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_samples,
                    num_workers=0)
```

Call `set_epoch` before each epoch to vary the deterministic per-image
augmentation. The example uses `num_workers=0`; persistent worker copies need
their epoch synchronized separately. Seed the DataLoader's shuffle generator
separately for reproducible ordering. `rare_count=5`, `10` or `15` selects the
corresponding recorded training subset. Validation cannot enable augmentation.
For a model, explicitly adapt image channels, dtype, normalization and resizing,
and transform annotations consistently. The loader itself preserves dimensions.

Training augmentation samples rotation up to 5 degrees, brightness multiplier
0.8–1.2 and Gaussian noise standard deviation 5 on a uint8 scale. Blur is
available through `augmentation_settings={"blur_sigma": 1.0}` and defaults to
zero. Rotations use median-intensity padding and can crop edge content. Boxes
are transformed enclosing rectangles, while masks use nearest-neighbor sampling.
These settings are illustrative and need validation against real factory data.
They are applied only when this loader is explicitly used for training.
