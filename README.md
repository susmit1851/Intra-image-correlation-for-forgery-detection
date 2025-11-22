# Intra-image-correlation-for-forgery-detection

Project for detecting manipulated regions within images using an intra-image correlation approach. This repository contains a PyTorch implementation that trains an instance-based model (using a SegFormer backbone) to predict forged regions as instance masks and aggregate them to a foreground forgery mask.

**Contents**
- **`train.py`**: Training script and dataset class used for experiments.
- **`model.py`**: Model definition (`CmfdInstanceModel`) using a SegFormer backbone and a small transformer decoder.
- **`mapping.py`**: Utilities to convert instance outputs into a final foreground mask (`instance_to_foreground`).
- **`loss.py`**: Loss definitions used during training.
- **`utils.py`**: Common imports and helper utilities.
- **`checkpoints/`**: Directory where model weights are saved during training.

**Requirements**
- Python 3.8+
- PyTorch (tested with 1.12+)
- torchvision
- transformers
- numpy
- Pillow
- scikit-learn
- tqdm

Install common dependencies (adjust versions as needed):

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision transformers numpy pillow scikit-learn tqdm
```

If you prefer a `requirements.txt`, create one with the packages listed above.

Data layout and expectations
- The training script `train.py` sets `DATASET_ROOT = "."` by default. That means it expects the dataset to be placed one directory above this repository root. You can either move your dataset accordingly or modify `DATASET_ROOT` in `train.py` to point to your dataset location.
- Expected directory structure (relative to `DATASET_ROOT`):

```
train_images/
	authentic/       # authentic images (PNG)
	forged/          # forged images (PNG)
train_masks/       # per-image mask files as NumPy .npy files
									 # mask filename must match image basename (e.g. image01.png -> image01.npy)
```

- Mask format: masks are loaded from `.npy` files. The code accepts 2D masks or 3D masks (channels-first); if multiple channels exist the code combines them via logical OR. Masks are resized to the training `image_size` using nearest-neighbor interpolation.

Training
- Basic training run (from repository root):

```bash
python train.py
```

- Defaults in `train.py`:
	- `image_size`: 256
	- `batch_size`: 4
	- `EPOCHS`: 25
	- Optimizer: `AdamW` with lr=3e-5
	- Learning-rate scheduler: `CosineAnnealingLR` (T_max=10)
	- Model: `CmfdInstanceModel(num_queries=5)` (see `model.py`)

- Checkpoints: Model weights are saved each epoch to the `checkpoints` directory as `cmfd_epoch_{epoch}.pth`.

Validation and metrics
- During training `train.py` computes Dice and IoU on the validation split. Predictions are formed by converting instance outputs to a foreground mask using `mapping.instance_to_foreground`.

Results

Validation (example run):

| Metric | Value |
|---|---:|
| `val_dice_pos` | 0.7873 |
| `val_dice_all` | 0.7395 |
| `val_iou_pos`  | 0.6492 |
| `val_iou_all`  | 0.5867 |

Loss plot: `plot/loss_plot.png`

Inline plot

![Training and Validation Loss](plot/loss_plot.png)

Inference (example)
The repository does not include a dedicated inference script. A minimal example to load a checkpoint and run inference on an image is provided earlier in this document; adapt that snippet for your use.

Checkpoint files
- Example checkpoint files found in `checkpoints/` follow the naming pattern: `cmfd_epoch_{epoch}.pth`. Use these to resume training or run inference. Loading is done with `model.load_state_dict(torch.load(...))` as in the example above.
- Example checkpoint files found in `checkpoints/` follow the naming pattern: `cmfd_epoch_{epoch}.pth`. Use these to resume training or run inference. Loading is done with `model.load_state_dict(torch.load(...))` as in the example above.





