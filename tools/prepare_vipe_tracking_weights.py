"""Populate tracking weights on an internet-connected CPU host, atomically."""
import os
from pathlib import Path
import urllib.request

import gdown
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1] / "local/vipe-cache"
assets = {
    "sam/sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "checkpoints/groundingdino_swint_ogc.pth": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    "aot/R50_DeAOTL_PRE_YTB_DAV.pth": None,
}
for name, url in assets.items():
    target = ROOT / "torch/hub" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"EXISTS {target}", flush=True)
        continue
    temporary = target.with_suffix(".download")
    print(f"Downloading {name}", flush=True)
    if url:
        urllib.request.urlretrieve(url, temporary)
    else:
        result = gdown.download(id="1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ", output=str(temporary))
        if not result:
            raise RuntimeError("AOT download failed")
    if temporary.stat().st_size < 1_000_000:
        raise RuntimeError(f"Invalid download: {temporary}")
    os.replace(temporary, target)
    print(f"READY {target} ({target.stat().st_size} bytes)", flush=True)
snapshot_download("bert-base-uncased", cache_dir=ROOT / "huggingface/hub",
                  allow_patterns=["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
print("Tracking downloads complete", flush=True)
