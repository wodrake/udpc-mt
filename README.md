# UDPC-MT

**Uncertainty-Decoupled Prototype Consistency with Mean Teacher for semi-supervised medical image segmentation.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

This repository is the code release for **Part I of a master's thesis**: a semi-supervised
segmentation framework that decides *where* consistency should be enforced and *where* it
should not, instead of applying a uniform consistency loss over the whole pseudo-labelled image.

> 中文说明见 [README.zh-CN.md](README.zh-CN.md)。

The quantitative results are **not published in this repository** — they are reported in the
thesis. What is published here is the method, the training/evaluation code, and the exact
experimental protocol needed to reproduce the setup on your own data.

---

## Method

UDPC-MT extends a standard Mean Teacher with two coupled ideas.

**1. Uncertainty-decoupled consistency.**
The teacher runs several stochastic forward passes (MC Dropout, 8 passes) and the entropy of
the mean probability map becomes an uncertainty map. The map is thresholded into a
low-uncertainty region and a high-uncertainty region, and the two regions are treated
differently:

- **Low uncertainty** — the teacher's pseudo-label is trustworthy, so the student is pulled
  towards it with a masked pixel-wise MSE consistency loss.
- **High uncertainty** — the pseudo-label is likely wrong, so pixel-wise consistency is
  *disabled*. Enforcing it here is what makes ordinary Mean Teacher stall at the lesion
  boundary; the boundary is exactly where the teacher disagrees with itself.

**2. Prototype branch for the discarded region.**
The region thrown away by consistency is not thrown away entirely. A class-wise prototype
memory stores an EMA of teacher decoder-head features for `background` and `lesion`
(initialized from the first valid batch, updated with GT masks on labeled data and
low-uncertainty pseudo-labels on unlabeled data). Student pixels in the *high*-uncertainty
region are then trained with a simplified InfoNCE against those prototypes, which gives the
network a coarse "is this pixel closer to lesion or background" signal where a per-pixel
target would be unreliable.

The total objective is

```
L = L_sup (BCE + Dice)  +  w_c · L_consistency(low-uncertainty)  +  w_p · L_prototype(high-uncertainty)
```

A short walk-through of the data flow, module by module, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation protocol

The thesis benchmark is set up as follows; the code in `configs/` and the per-method recipes
under the repository root implement it.

- **Dataset** — BUSI breast ultrasound, 2D, 256×256, min–max normalization.
- **Split** — 544 train / 118 val / 118 test, frozen once and shared by every run.
  Seed 42 drives the split, the labeled/unlabeled assignment and the first training repeat.
- **Label scarcity** — 20% of the training set is labeled (109 images), the rest (435) is
  unlabeled; unlabeled masks are never exposed to the training loop.
- **Budget** — 220 epochs, batch 8 (4 labeled), Adam-style optimizer at 1e-3, EMA decay 0.995,
  20-epoch consistency ramp-up.
- **Repeats** — 10 training seeds per method (42–51) with the split held fixed, so the
  reported variance is training variance, not split variance.
- **Compared methods** — UNet, Mean Teacher, UAMT, CCT, CPS, EM, URPC, MCNet, SCP, PCCS, SLC,
  and UDPC-MT. Each baseline keeps the backbone, optimizer and iteration budget of its
  reference implementation, so this is a protocol-level comparison of published recipes, not
  an ablation under one shared recipe.

## Repository layout

```
udpc-mt/
├── train.py / val.py / infer.py      # root entry points (config-driven)
├── experiment_logger.py              # local + optional W&B logging
├── configs/                          # dataset and method configs (BUSI / ACDC / BML / TN3K)
├── datasets/                         # dataset adapters: busi, acdc, bml, tn3k, semi_split
├── models/                           # UNet, EMA teacher, prototype memory
├── losses/                           # supervised, consistency, prototype, total
├── trainers/                         # baseline and mean-teacher training loops
├── udpc_mt/                          # the full UDPC-MT package (model / engine / data)
├── utils/                            # uncertainty, metrics, checkpointing
├── scripts/                          # data preparation, smoke tests, compatibility entry points
├── tests/                            # pytest suite
├── docs/                             # architecture notes, runbooks
└── *.yaml / launch_*.sh              # dated experiment recipes from the thesis runs
```

The dated `*.yaml`/`launch_*.sh` files at the repository root are the actual recipes used for
the thesis runs, kept verbatim for reproducibility. They contain absolute cluster paths
(`/data1/houshengdi/udpc-mt/...`) and must be edited before reuse on another machine.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install a CUDA-matching torch wheel first if needed

python scripts/prepare_busi.py           # -> data/processed/busi/{images,masks,metadata.csv}

# full UDPC-MT (220 epochs)
python train.py --config configs/busi_udpcmt.yaml

# 1-epoch smoke run
python train.py --config configs/busi_udpcmt_smoke.yaml

# evaluate / infer
python val.py   --config configs/busi_udpcmt.yaml --ckpt outputs/runs/<run>/checkpoints/best.pt --split test
python infer.py --config configs/busi_udpcmt.yaml --ckpt outputs/runs/<run>/checkpoints/best.pt --input <image.png>
```

Every run writes `config_resolved.yaml`, `splits.json`, `train.log`, `history.json`,
`summary.json`, `checkpoints/best.pt` and `checkpoints/last.pt` under
`outputs/runs/<run_name>/`. Training is resumable:

```bash
python train.py --resume outputs/runs/<run>/checkpoints/last.pt --override train.epochs=260
```

### Modes

| `mode` | What runs |
|--------|-----------|
| `baseline` | supervised UNet, BCE + Dice on labeled data |
| `mean_teacher` | EMA teacher + masked MSE consistency on unlabeled data |
| `udpc_mt` | mean teacher + uncertainty decoupling + prototype contrastive branch |

### Main hyper-parameters

| Key | Value in the thesis runs | Meaning |
|-----|--------------------------|---------|
| `train.ema_decay` | 0.995 | teacher EMA |
| `train.rampup_epochs` | 20 | consistency weight ramp-up |
| `uncertainty.mc_passes` | 8 | stochastic teacher forwards |
| `uncertainty.threshold` | 0.15 | entropy threshold splitting low/high |
| `loss.prototype_weight` | 0.08 | weight of the prototype term |
| `prototype.min_pixels` | 16 | minimum region size to emit a contrastive term |
| `prototype.temperature` | 0.1 | InfoNCE temperature |
| `dataset.image_size` | 256 | input resolution |

## Running your own benchmark

1. Prepare BUSI and build a frozen split with the sizes and ratios above
   (`split_seed` / `label_seed` in `dataset.*`).
2. For each method config and each seed, run `train.py --config <config>` with
   `seed` and `split_manifest` pointed at your frozen split; evaluate the best checkpoint on
   `test`.
3. Aggregate the per-run test Dice yourself. `outputs/runs/<run_name>/summary.json` holds the
   test metrics of a single run; nothing is aggregated across runs by the code.

## Limitations

- **2D only.** ACDC is handled as independent 2D slices; no 3D/2.5D volumetric model is
  implemented.
- **Artificial label scarcity.** BUSI is fully annotated; the 10%/20% regimes are simulated by
  hiding labels, which is more optimistic than real clinical scarcity.
- **Recipes differ across rows.** Each baseline keeps its reference implementation's recipe, so
  cross-method gaps must be read with the protocol in mind; a same-recipe ablation lives in the
  thesis, not in this repository.
- **Hyper-parameter tuning.** The UDPC-MT recipe was selected by a sweep on the same benchmark
  splits, so any margin over baselines is mildly optimistic.
- Datasets other than BUSI (ACDC / BML / TN3K / LAA1 / LAA2) are supported by the code but are
  not part of the headline setup.

## Citation

The thesis is in preparation. If you use this code, please cite it once the thesis is public;
in the meantime a link to this repository is enough.

## Acknowledgements

- BUSI: Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.
- Baseline implementations follow their original papers.

## License

[MIT](LICENSE).
