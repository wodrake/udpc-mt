# UDPC-MT 中文说明

**不确定性解耦的原型一致性半监督医学图像分割**，本仓库是硕士学位论文第一部分工作的代码。

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

> 具体实验数值不在本仓库公布，以论文正文为准。这里公开的是方法、训练/评估代码，以及复现实验设置所需的协议。

## 方法

标准 Mean Teacher 对所有无标注像素一视同仁地施加一致性约束，但在病灶边界处 teacher 自己都
不确定，强行拉齐反而会把 student 拉向错误目标。UDPC-MT 的做法是**按不确定性分区处理**：

1. **不确定性估计**：teacher 做 8 次 MC Dropout 前向，取平均概率图的熵作为不确定性图，按阈值
   切成低不确定区域与高不确定区域。
2. **解耦一致性**：低不确定区域保留 masked MSE 像素一致性；高不确定区域**关闭**像素一致性。
3. **原型分支**：被一致性丢弃的高不确定区域并不是直接扔掉，而是让 student 的像素特征去和
   类别原型（background / lesion）做简化 InfoNCE。原型由 teacher 的 decoder-head 特征以 EMA
   方式维护，标注数据用 GT 更新，无标注数据只用低不确定伪标签更新。

总损失：

```
L = L_sup (BCE + Dice) + w_c · L_一致性(低不确定区域) + w_p · L_原型对比(高不确定区域)
```

模块级的调用链见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 实验协议

- **数据**：BUSI 乳腺超声，2D，256×256，min–max 归一化。
- **划分**：train 544 / val 118 / test 118，一次性冻结，所有方法共用。
- **标注比例**：train 中 20% 有标注（109 张），其余 435 张作为无标注；无标注样本的 mask
  不会进入训练循环。
- **训练预算**：220 epoch，batch 8（其中有标注 batch 4），学习率 1e-3，EMA 0.995，
  一致性权重 20 epoch 斜坡上升。
- **重复**：每个方法跑 10 个训练种子（42–51），划分固定不变，因此方差是训练方差而非划分方差。
- **对比方法**：UNet、Mean Teacher、UAMT、CCT、CPS、EM、URPC、MCNet、SCP、PCCS、SLC，
  以及本文的 UDPC-MT。各 baseline 沿用其原实现的骨干、优化器与迭代预算，属于"公开配方之间"
  的对比，而非同一配方下的消融。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_busi.py                  # 准备 BUSI 到 data/processed/busi

python train.py --config configs/busi_udpcmt.yaml          # 完整训练
python train.py --config configs/busi_udpcmt_smoke.yaml    # 1 epoch 冒烟

python val.py   --config configs/busi_udpcmt.yaml --ckpt outputs/runs/<run>/checkpoints/best.pt --split test
python infer.py --config configs/busi_udpcmt.yaml --ckpt outputs/runs/<run>/checkpoints/best.pt --input <image.png>
```

三种训练模式：`baseline`（纯监督）、`mean_teacher`（EMA + 一致性）、`udpc_mt`（+ 不确定性
解耦 + 原型对比）。切换数据集只改 `dataset.*` 配置，已支持 BUSI / ACDC / BML / TN3K。

## 关键超参

| 配置项 | 取值 |
|--------|------|
| `train.ema_decay` | 0.995 |
| `train.rampup_epochs` | 20 |
| `uncertainty.mc_passes` | 8 |
| `uncertainty.threshold` | 0.15 |
| `loss.prototype_weight` | 0.08 |
| `prototype.min_pixels` | 16 |
| `prototype.temperature` | 0.1 |
| `dataset.image_size` | 256 |

## 已知限制

- 仅 2D；ACDC 按独立切片处理，无 3D/2.5D 版本。
- BUSI 本身全标注，10%/20% 是人为隐藏标签构造的，比真实临床标注稀缺更乐观。
- 各方法沿用各自原配方，跨方法差距需结合协议解读；同配方消融见论文正文。
- UDPC-MT 的超参是在同一批划分上扫出来的，因此相对 baseline 的差距略偏乐观。
- ACDC / BML / TN3K / LAA1 / LAA2 支持但不进入主实验设置。

## 许可证

[MIT](LICENSE)。
