# PROJECT_STRUCTURE

## Design Goal

Keep the repository easy to run, easy to extend, and easy to resume in a new session.

## Chosen Layout

The project keeps a package-centric structure instead of flattening all modules into the repository root:

```text
udpc-mt/
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
├── data/
├── datasets/
├── docs/
├── losses/
├── models/
├── outputs/
├── scripts/
├── tests/
├── trainers/
├── utils/
├── train.py
├── val.py
└── udpc_mt/
    ├── data/
    ├── engine/
    ├── models/
    └── utils/
```

## Why This Structure

- `datasets/` holds dataset readers, augmentation policies, and semi-supervised split logic.
- `models/`, `losses/`, `utils/`, `trainers/` hold the root-level supervised baseline stack.
- `scripts/` keeps CLI entrypoints separate from library code.
- `udpc_mt/` holds reusable Python modules.
- `configs/` keeps hyperparameters out of training code.
- `outputs/` centralizes generated artifacts.
- `docs/` stores recovery, design, and planning context.

## Module Mapping

- `datasets/`
  - BUSI manifest discovery
  - weak/strong augmentation policies
  - train/val/test split
  - labeled/unlabeled split
- `models/`
  - supervised UNet baseline
- `losses/`
  - BCE + Dice supervised loss
- `utils/`
  - segmentation metrics
  - checkpoint I/O
- `trainers/`
  - supervised trainer
- `udpc_mt/data/`
  - compatibility shim for the training code
- `udpc_mt/models/`
  - segmentation backbone
  - feature embedding heads
- `udpc_mt/engine/`
  - training loop
  - evaluation
  - losses
  - prototype memory
- `udpc_mt/utils/`
  - config loading
  - metrics
  - seeding
  - filesystem helpers

## Default Technical Choices

- Dataset: BUSI binary segmentation
- Input modality: grayscale breast ultrasound
- Backbone: UNet
- Supervised loss: BCE + Dice
- Consistency loss: MSE
- Uncertainty estimation: MC Dropout
- Prototype memory: class-wise EMA prototype
- Semi-supervised framework: Mean Teacher
- Task formulation: lesion foreground vs background

## Current Artifact Policy

- training outputs: `outputs/runs/`
- prediction outputs: `outputs/predictions/`
- processed BUSI data: `data/processed/busi/`
- raw dataset mirror: `data/raw/`

Older artifacts under `runs/` and `predictions/` are legacy from the prototype stage.
