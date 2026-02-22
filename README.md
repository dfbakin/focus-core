# Classroom Group Engagement Recognition

Automatic recognition of group engagement levels from classroom video recordings, with post-session engagement timeline reports.

## Problem

Teachers cannot objectively assess collective student engagement during lessons. Manual observation is subjective and inconsistent.

## Approach

1. Reproduce OUC-CGE baseline (~98% accuracy with SLOW model)
2. Compare video architectures (I3D, SlowFast, SLOW, X3D)
3. Implement pose/landmark models
4. Validate generalization on DIPSER
5. Build inference pipeline and report generation

## Datasets

| Dataset | Setting | Labels | Duration | Use |
|---------|---------|--------|----------|-----|
| **OUC-CGE** | Classroom | 3-level group engagement | 12.8h | Primary benchmark |
| **DIPSER** | Classroom | 5-level individual attention | 51.3h | Cross-dataset validation |

## Project Structure

```
focus-core/
├── configs/                    # Hydra YAML config overrides
│   └── experiment/             #   composed experiment presets
│
├── src/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation entry point
│   │
│   ├── conf/                   # Structured config dataclasses
│   │   ├── config.py           #   root config + Hydra registration
│   │   ├── data.py             #   dataset configs
│   │   ├── model.py            #   model configs
│   │   └── trainer.py          #   trainer configs
│   │
│   ├── data/                   # Data loading
│   │   ├── video_dataset.py    #   abstract base class
│   │   ├── ouc_cge.py          #   OUC-CGE dataset (to implement)
│   │   ├── dipser.py           #   DIPSER dataset (to implement)
│   │   ├── datamodule.py       #   Lightning DataModule
│   │   └── transforms.py       #   video transforms (to implement)
│   │
│   ├── models/                 # Model definitions
│   │   ├── base.py             #   base LightningModule for video classification
│   │   ├── factory.py          #   model registry + factory
│   │   └── backbones/          #   video model wrappers
│   │       └── pytorchvideo.py #     PyTorchVideo models (to implement)
│   │
│   ├── features/               # Feature extraction (Phase 3-4)
│   │   ├── pose.py             #   body pose (MediaPipe BlazePose)
│   │   └── landmarks.py        #   facial landmarks (MediaPipe Face Mesh)
│   │
│   ├── fusion/                 # Multimodal fusion (Phase 4)
│   │   └── strategies.py       #   early/late/attention fusion
│   │
│   ├── metrics/                # Evaluation metrics
│   ├── training/               # Training utilities
│   │   └── callbacks.py        #   MLflow logging, early stopping
│   ├── utils/                  # General utilities
│   │   └── logging.py          #   MLflow helpers
│   │
│   ├── inference/              # Inference pipeline (Phase 6)
│   │   └── pipeline.py         #   end-to-end video inference
│   │
│   └── reporting/              # Report generation (Phase 7)
│       └── engagement_report.py
│
├── scripts/                    # Utility scripts
├── docs/                       # Project documentation
│   ├── implementation_plan.md
│   └── literature_review.md
│
├── pyproject.toml
├── requirements.txt
├── Dockerfile                  # Vast.ai GPU container
├── docker-compose.yml
└── .dvc/                       # Data version control
```

## Setup

### Local

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt

# Or with Poetry
poetry install
```

## Usage

```bash
# Train with defaults (SLOW on OUC-CGE)
python -m src.train

# Override model
python -m src.train model=slowfast

# Override dataset
python -m src.train data=dipser model=i3d

# Use experiment preset
python -m src.train +experiment=baseline_ouc_cge

# Multi-run sweep
python -m src.train --multirun model=slow,slowfast

# Evaluate checkpoint
python -m src.evaluate +checkpoint_path=outputs/.../best.ckpt model=slow
```

## Adding New Components

### New Model

1. Write a factory function in `src/models/backbones/` that returns `nn.Module`
2. Decorate it with `@register_model("your_model_name")`
3. Add a config dataclass in `src/conf/model.py`
4. Register it in `src/conf/config.py`
5. (Optional) Add an experiment preset in `configs/experiment/`

### New Dataset

1. Subclass `VideoDataset` in `src/data/`
2. Add a config dataclass in `src/conf/data.py`
3. Register it in `src/conf/config.py`
4. Wire it in `src/train.py::build_datamodule()`
5. (Optional) Add an experiment preset in `configs/experiment/`

## Experiment Tracking

MLflow is used for experiment tracking. Start the UI:

```bash
mlflow ui --port 5000
```

## Data Versioning

DVC is used for large dataset management:

```bash
dvc pull    # download data
dvc push    # upload data
```
