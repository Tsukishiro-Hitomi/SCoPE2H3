"""Fail before GPU inference if the pose pipeline's offline assets are missing."""
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, BertModel

root = Path(os.environ["TORCH_HOME"]) / "hub"
names = ["sam/sam_vit_b_01ec64.pth", "aot/R50_DeAOTL_PRE_YTB_DAV.pth",
         "checkpoints/groundingdino_swint_ogc.pth", "droid_slam/droid.pth",
         "geocalib/pinhole.tar"]
missing = [str(root / name) for name in names if not (root / name).is_file()]
if missing:
    raise SystemExit("Missing offline weights:\n" + "\n".join(missing))
for name in names:
    path = root / name
    # These are trusted upstream checkpoints; validate deserialization on CPU.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print(f"LOAD OK {path}", flush=True)
    del checkpoint
AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
del model
from huggingface_hub import hf_hub_download
hf_hub_download("Ruicheng/moge-2-vitl", "model.pt", local_files_only=True)
print("Offline asset preflight passed", flush=True)
