#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


CONFIG_TEMPLATE = """project_name: {project_name}
mode: udpc_mt
seed: 42
device: cuda:0
output_root: outputs/runs

dataset:
  name: busi
  root: data/processed/busi
  image_size: 256
  val_fraction: 0.0
  test_fraction: 0.0
  labeled_fraction: {ratio_value}
  include_normal_as_unlabeled: false
  num_workers: 0
  sample_limit: null
  split_source: official
  split_files:
    labeled: {split_dir}/labeled.txt
    unlabeled: {split_dir}/unlabeled.txt
    val: {split_dir}/val.txt
    test: {split_dir}/test.txt
  task_type: binary
  augmentation_profile: ultrasound
  preprocessing:
    normalization: minmax

train:
  epochs: 220
  batch_size: 8
  labeled_batch_size: 4
  max_steps_per_epoch: null
  lr: 0.001
  weight_decay: 0.0001
  ema_decay: 0.995
  rampup_epochs: 20
  log_interval: 10
  save_best_metric: dice

model:
  in_channels: 1
  num_classes: 1
  base_channels: 32
  prototype_dim: 64
  dropout: 0.2

loss:
  bce_weight: 0.5
  dice_weight: 0.5
  consistency_weight: 1.0
  prototype_weight: 0.08

uncertainty:
  enabled: true
  mc_passes: 8
  threshold: 0.15

prototype:
  enabled: true
  num_classes: 2
  feature_source: decoder_head
  init_strategy: first_valid_batch
  normalize: true
  temperature: 0.1
  momentum: 0.95
  min_pixels: 16

evaluation:
  metrics:
  - dice
  - iou
  - precision
  - recall
  - specificity

wandb:
  enabled: true
  project: udpc-mt
  run_name: {wandb_run_name}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a BUSI UDPC-MT explicit-split YAML config.")
    parser.add_argument("--output", required=True, help="Path to the YAML file to write.")
    parser.add_argument("--project-name", required=True, help="project_name field.")
    parser.add_argument("--wandb-run-name", required=True, help="wandb.run_name field.")
    parser.add_argument("--ratio-value", type=float, required=True, help="Labeled ratio in [0, 1].")
    parser.add_argument("--split-dir", required=True, help="Split directory relative to project root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = CONFIG_TEMPLATE.format(
        project_name=args.project_name,
        wandb_run_name=args.wandb_run_name,
        ratio_value=args.ratio_value,
        split_dir=args.split_dir,
    )
    output_path.write_text(content, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
