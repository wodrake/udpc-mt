# MULTI_DATASET_SERVER_RUNBOOK

## 目标

你已经在服务器上跑通了 BUSI。接下来如果要继续跑 `BML` 和 `ACDC`，当前仓库已经具备核心代码，不需要修改 Python 源码，重点是：

- 准备对的数据目录
- 选对配置文件
- 在服务器上新建你自己的 server config
- 先跑 smoke，再跑完整训练

## 这次仓库里新增了什么

这次同步到服务器后，你会多出以下文件：

- `scripts/prepare_acdc.py`
- `scripts/prepare_bml.py`
- `configs/acdc_udpcmt_smoke.yaml`
- `configs/bml_udpcmt_smoke.yaml`
- `configs/acdc_udpcmt_server.example.yaml`
- `configs/bml_udpcmt_server.example.yaml`
- `docs/MULTI_DATASET_SERVER_RUNBOOK.md`

其中：

- `*_smoke.yaml` 用来先验证链路能否启动
- `*_server.example.yaml` 用来给你在服务器上复制成自己的正式配置
- `prepare_acdc.py` 用来直接下载 ACDC 公开镜像到 `data/processed/acdc`
- `prepare_bml.py` 用来把你拿到的 BML 目录、压缩包或 URL 整理成 `data/processed/bml`

如果你手里是超声视频帧数据而不是 BUSI 单图，这次也可以直接用：

- `scripts/prepare_us_video_dataset.py`
- `configs/custom_us_busi_like_udpcmt_server.example.yaml`

## 如果服务器上还没有数据集

### ACDC

可以直接在服务器上下载：

```bash
python scripts/prepare_acdc.py
```

下载完成后，默认目录就是：

```text
data/processed/acdc/
```

### BML

这里要区分清楚：

- 当前仓库没有内置一个默认的公开 BML 下载地址
- 你需要先拿到 BML 数据本体，或者拿到一个可直接下载压缩包的 URL

拿到数据以后，再用下面任一方式整理到项目目录：

```bash
python scripts/prepare_bml.py --input-dir /path/to/your/bml_source
python scripts/prepare_bml.py --archive-path /path/to/bml.zip
python scripts/prepare_bml.py --source-url https://example.com/bml.zip
```

### 自定义超声视频数据

如果你的目录类似：

```text
/your/us_dataset_root/
├── LAA1/
│   ├── videos/{train,val,test}/*.npy
│   └── annotations/{train,val,test}/*.npz
└── LAA2/
    ├── videos/{train,val,test}/*.npy
    └── annotations/{train,val,test}/*.npz
```

先转成 BUSI 风格：

```bash
python scripts/prepare_us_video_dataset.py \
  --input-root /your/us_dataset_root \
  --output-root data/processed/us_busi_like \
  --datasets LAA1 LAA2 \
  --pairing-mode auto \
  --labeled-ratios 0.1 0.2
```

转换后会得到：

```text
data/processed/us_busi_like/laa1/
├── images/
├── masks/
├── metadata.csv
├── summary.json
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── ratio_10/
    │   ├── labeled.txt
    │   ├── unlabeled.txt
    │   ├── val.txt
    │   └── test.txt
    └── ratio_20/
        ├── labeled.txt
        ├── unlabeled.txt
        ├── val.txt
        └── test.txt
```

关键点：

- 默认只导出有标注的关键帧，不会把整段视频全部展开
- `train / val / test` 保留原始目录划分
- `labeled / unlabeled` 按视频分组切，不按帧随机切
- 如果完整数据的 `video` 和 `annotation` 文件名本身一一对应，保留 `--pairing-mode auto`
- 如果文件名不对应但目录顺序本身是对齐的，可以显式改成 `--pairing-mode sorted`

## 你在服务器上需要新建什么文件

推荐你不要直接改仓库里现成的 `configs/acdc_udpcmt.yaml` 和 `configs/bml_udpcmt.yaml`，而是在服务器上新建你自己的配置文件：

```bash
cp configs/acdc_udpcmt_server.example.yaml configs/acdc_udpcmt_server.yaml
cp configs/bml_udpcmt_server.example.yaml configs/bml_udpcmt_server.yaml
```

然后只改这两个新文件。

## 你在服务器上需要修改什么字段

### ACDC

编辑 `configs/acdc_udpcmt_server.yaml`，至少改这些字段：

- `device`
  - 例如 `cuda:0`
- `dataset.root`
  - 改成你的 ACDC 数据根目录
- `dataset.num_workers`
  - 常见取值 `4 / 8`
- `train.batch_size`
  - 显存不够就减小
- `train.labeled_batch_size`
  - 一般不要超过 `train.batch_size`
- `output_root`
  - 如果你希望结果落到别的磁盘，改这里

### BML

编辑 `configs/bml_udpcmt_server.yaml`，至少改这些字段：

- `device`
- `dataset.root`
- `dataset.num_workers`
- `train.batch_size`
- `train.labeled_batch_size`
- `output_root`

## 数据目录要求

### ACDC 目录

当前 ACDC 适配器期待原始 patient 目录风格：

```text
/your/acdc/root/
├── patient001/
│   ├── Info.cfg
│   ├── patient001_frame01.nii.gz
│   ├── patient001_frame01_gt.nii.gz
│   ├── patient001_frame02.nii.gz
│   └── patient001_frame02_gt.nii.gz
├── patient002/
└── ...
```

关键点：

- 必须有 `patientXXX/`
- 必须有 `Info.cfg`
- 必须成对存在 `*_frameXX.nii.gz` 和 `*_frameXX_gt.nii.gz`
- 当前实现会自动把 3D volume 展成 `2D single-slice`
- 当前 split 是按病人分组，不是按 slice 随机切

### BML 目录

当前 BML 支持两种风格，二选一。

第一种：`metadata.csv`

```text
/your/bml/root/
├── metadata.csv
├── images/
└── masks/
```

第二种：显式 split 目录

```text
/your/bml/root/
├── train/
│   ├── image/
│   └── mask/
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

关键点：

- 文件名最好一一对应
- 如果 mask 文件名是 `<stem>_mask.png` 也能识别
- 如果你只有 `train/` 和 `val/`，当前实现会复用 `val` 作为 `test`

## 推荐执行顺序

### 1. 先同步这些文件到服务器

最少要同步：

- `scripts/prepare_acdc.py`
- `scripts/prepare_bml.py`
- `scripts/prepare_us_video_dataset.py`
- `configs/acdc_udpcmt_smoke.yaml`
- `configs/bml_udpcmt_smoke.yaml`
- `configs/acdc_udpcmt_server.example.yaml`
- `configs/bml_udpcmt_server.example.yaml`
- `configs/custom_us_busi_like_udpcmt_server.example.yaml`
- `docs/MULTI_DATASET_SERVER_RUNBOOK.md`
- `README.md`

如果你是整体同步仓库，这一步自然就包含了。

### 2. 如果缺数据，先准备数据

#### ACDC

```bash
python scripts/prepare_acdc.py
```

#### BML

```bash
python scripts/prepare_bml.py --input-dir /path/to/your/bml_source
```

### 3. 先跑 smoke，确认链路没问题

#### ACDC smoke

如果你想不新建正式配置，先直接用 override 快速验证：

```bash
python train.py \
  --config configs/acdc_udpcmt_smoke.yaml \
  --override device=cuda:0 \
  --override dataset.root=/your/acdc/root \
  --override dataset.num_workers=4
```

这个配置只会：

- 跑 `1` 个 epoch
- 每个 epoch 最多跑 `1` 个 train step
- 主要用途是确认 dataloader、model、checkpoint、日志能不能走通

#### BML smoke

```bash
python train.py \
  --config configs/bml_udpcmt_smoke.yaml \
  --override device=cuda:0 \
  --override dataset.root=/your/bml/root \
  --override dataset.num_workers=4
```

### 4. 再新建正式 server config

#### ACDC

```bash
cp configs/acdc_udpcmt_server.example.yaml configs/acdc_udpcmt_server.yaml
```

然后编辑 `configs/acdc_udpcmt_server.yaml`。

#### BML

```bash
cp configs/bml_udpcmt_server.example.yaml configs/bml_udpcmt_server.yaml
```

然后编辑 `configs/bml_udpcmt_server.yaml`。

### 5. 跑正式训练

#### ACDC 正式训练

```bash
python train.py --config configs/acdc_udpcmt_server.yaml
```

#### BML 正式训练

```bash
python train.py --config configs/bml_udpcmt_server.yaml
```

### 6. 跑验证

训练结束后，拿 `best.pt` 跑验证。

#### ACDC 验证

```bash
python val.py \
  --config configs/acdc_udpcmt_server.yaml \
  --ckpt outputs/runs/<acdc_run_name>/checkpoints/best.pt \
  --split test
```

#### BML 验证

```bash
python val.py \
  --config configs/bml_udpcmt_server.yaml \
  --ckpt outputs/runs/<bml_run_name>/checkpoints/best.pt \
  --split test
```

### 7. 跑推理

#### BML 推理

BML 是图片型数据，可以直接用现有 `infer.py`：

```bash
python infer.py \
  --config configs/bml_udpcmt_server.yaml \
  --ckpt outputs/runs/<bml_run_name>/checkpoints/best.pt \
  --input /your/bml/root/test/image \
  --gt-dir /your/bml/root/test/mask \
  --output-dir outputs/predictions/bml_test
```

#### ACDC 推理

这里要特别注意：

- 当前 `infer.py` 是面向 `png/jpg/tif` 这类 2D 图片的
- 当前不直接读取 ACDC 的原始 `.nii.gz` volume

所以对 ACDC 来说：

- 训练和验证已经支持
- `val.py` 是当前更合适的评估入口
- 如果你需要 ACDC 的原始 NIfTI 推理导出，建议下一步单独加一个专门脚本，而不是直接套现在的 `infer.py`

## 不想新建 server config 时的替代方案

如果你只是临时跑一次，也可以完全不新建 `*_server.yaml`，直接用 override：

### ACDC

```bash
python train.py \
  --config configs/acdc_udpcmt.yaml \
  --override device=cuda:0 \
  --override dataset.root=/your/acdc/root \
  --override dataset.num_workers=4 \
  --override train.batch_size=4 \
  --override train.labeled_batch_size=2
```

### BML

```bash
python train.py \
  --config configs/bml_udpcmt.yaml \
  --override device=cuda:0 \
  --override dataset.root=/your/bml/root \
  --override dataset.num_workers=4 \
  --override train.batch_size=4 \
  --override train.labeled_batch_size=2
```

这种方式的优点是不用新建文件，缺点是命令长，而且不利于你后面复现。

## 推荐你优先改哪些参数

如果服务器显存一般，先从这几个参数下手：

- `train.batch_size`
- `train.labeled_batch_size`
- `dataset.num_workers`
- `dataset.image_size`
- `uncertainty.mc_passes`

一个保守起点：

- `train.batch_size: 4`
- `train.labeled_batch_size: 2`
- `dataset.num_workers: 4`

如果还是爆显存，再降：

- `train.batch_size: 2`
- `train.labeled_batch_size: 1`

## 恢复训练

### ACDC

```bash
python train.py \
  --resume outputs/runs/<acdc_run_name>/checkpoints/last.pt \
  --override train.epochs=80
```

### BML

```bash
python train.py \
  --resume outputs/runs/<bml_run_name>/checkpoints/last.pt \
  --override train.epochs=50
```

## 最后结论

要跑 `BML` 和 `ACDC`，你现在不需要改 Python 代码。推荐做法是：

1. 同步这次新增的配置和文档到服务器
2. 如果服务器没数据，先用 `prepare_acdc.py` 或 `prepare_bml.py` 准备数据
3. 再用 `*_smoke.yaml` 跑 1 次最小链路
4. 再在服务器新建 `*_server.yaml`
5. 修改 `device`、`dataset.root`、`num_workers`、batch size
6. 再跑正式训练和验证

如果后面你还想让我继续补，我建议下一步补的是：`ACDC` 的原生 NIfTI 推理脚本。
