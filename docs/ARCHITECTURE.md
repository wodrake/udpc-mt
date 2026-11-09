# Architecture

How a run flows from a YAML file to a trained checkpoint. The point of this document is that a
new session can pick the project up without re-reading the whole codebase.

## Entry points

| Script | Purpose |
|--------|---------|
| `train.py` | training for all three modes, resume support |
| `val.py` | evaluate a checkpoint on `val` or `test` |
| `infer.py` | single-image or folder inference, writes overlays and uncertainty maps |
| `scripts/train.py` / `scripts/validate.py` / `scripts/infer.py` | thin compatibility wrappers around the root scripts |
| `scripts/smoke_test.py` | 1-epoch end-to-end checks for each mode |

## Dispatch

```
config.yaml (or --resume <ckpt>)
   └─ mode: baseline | mean_teacher | udpc_mt
        ├─ baseline / mean_teacher -> trainers/trainer.py
        └─ udpc_mt                 -> udpc_mt/engine/trainer.py
```

The config is resolved once, snapshotted to `config_resolved.yaml`, and every default can be
overridden on the command line with `--override key.subkey=value`.

## Data layer

| File | Role |
|------|------|
| `datasets/base.py` | dataset adapter abstraction and the common dataset class |
| `datasets/factory.py` | builds train/val/test dataloaders from config |
| `datasets/busi.py`, `acdc.py`, `bml.py`, `tn3k.py` | per-dataset adapters (layout differences live *only* here) |
| `datasets/semi_split.py` | train/val/test split and labeled/unlabeled partition |
| `datasets/transforms.py` | weak augmentation (teacher) and strong augmentation (student) |

A labeled batch yields `(image, mask)`. An unlabeled batch yields `(teacher_image, student_image)`
— weak view for the teacher, strong view for the student. Unlabeled samples never expose their
mask to the training loop, even when a ground truth exists on disk.

## Model layer

| File | Role |
|------|------|
| `models/unet.py` | root-level UNet, returns logits only |
| `udpc_mt/models/unet.py` | UNet used by `udpc_mt`, returns logits **and** decoder-head embeddings |
| `models/ema_teacher.py` | teacher copy with EMA update |
| `models/prototype_memory.py` | class-wise EMA prototype bank |

The prototype bank is keyed on `background` / `lesion`. Labeled samples update it from ground
truth; unlabeled samples update it from *low-uncertainty* pseudo-labels only. Features are taken
from the decoder head because that is the resolution at which pixel labels and masks line up.

## Loss layer

| File | Role |
|------|------|
| `losses/supervised.py` | BCE + Dice |
| `losses/consistency.py` | masked MSE on the low-uncertainty region |
| `losses/prototype.py` | simplified InfoNCE from high-uncertainty student pixels to class prototypes |
| `losses/total_loss.py` | weighted sum of the three terms |

## One `udpc_mt` training step

1. Sample a labeled batch and an unlabeled batch.
2. `student(labeled_images)` → supervised loss.
3. `student(student_images)` → logits + embeddings for the strong view.
4. `teacher(teacher_images)` with MC Dropout (8 passes) → mean probability, pseudo-label, entropy map.
5. Threshold the entropy map into `low_uncertainty_mask` / `high_uncertainty_mask`.
6. Masked MSE consistency **only** on the low-uncertainty region.
7. Prototype contrastive loss from student embeddings **only** on the high-uncertainty region.
8. `total = supervised + w_c · consistency + w_p · prototype`, backward, optimizer step.
9. EMA-update the teacher.
10. Update the prototype memory with teacher features under GT / low-uncertainty pseudo-labels.

## Per-epoch bookkeeping

After every epoch the trainer validates, appends to `history.json` and writes
`checkpoints/best.pt` / `checkpoints/last.pt`. When training ends it reloads the best checkpoint,
runs `test` once, and writes `summary.json` and `train.log`. Everything lands in
`outputs/runs/<run_name>/`.

## Uncertainty

`utils/uncertainty.py` implements MC Dropout: N stochastic forwards with dropout enabled, mean
probability map, and the entropy of that mean as the uncertainty signal. It returns the binary
low/high masks consumed by the losses. There is no epistemic/aleatoric decomposition — a single
entropy threshold is deliberately used, and `uncertainty.threshold` is the knob that trades
consistency coverage against pseudo-label purity.
