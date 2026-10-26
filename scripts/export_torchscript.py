"""Exporting SLOW r50 from a checkpoint to TorchScript for CPU inference"""

import torch
from src.models.factory import create_model
from src.models.base import VideoClassificationModule
from src.models.backbones import *

CKPT = "outputs/baseline_ouc_cge/2026-05-31_15-35-39/checkpoints/epoch=32-val/accuracy=1.0000.ckpt"
OUT = "engagement_slow_r50.ts"

backbone = create_model("slow_r50", num_classes=3, pretrained=False)
module = VideoClassificationModule.load_from_checkpoint(
    checkpoint_path=CKPT,
    model=backbone,
)

module.eval()
module.cpu()

model = module.model
model.eval()

example = torch.randn(1, 3, 8, 224, 224)

with torch.no_grad():
    traced = torch.jit.trace(model, example)
traced.save(OUT)

with torch.no_grad():
    a = model(example)
    b = traced(example)
    print("coincidence :", torch.allclose(a, b, atol=1e-5))
    print("shape of output:", a.shape)
print("download in", OUT)