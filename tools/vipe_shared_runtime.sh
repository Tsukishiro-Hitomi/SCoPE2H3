#!/usr/bin/env bash
set -euo pipefail
PROJECT=/inspire/hdd/project/wuliqifa/public/zxr/scope2h3
exec "${PROJECT}/local/python312/bin/python3.12" -c '
import os, runpy, site, sys
root = "/inspire/hdd/project/wuliqifa/public/zxr/scope2h3/local/third_party/vipe/.venv"
site.addsitedir(root + "/lib/python3.12/site-packages")
if os.environ.get("HF_HUB_OFFLINE") == "1":
    import torch, gdown
    def offline_download(*args, **kwargs):
        raise RuntimeError("Offline ViPE: missing cached asset; prepare weights on the CPU host before running inference")
    torch.hub.download_url_to_file = offline_download
    gdown.download = offline_download
sys.argv[0] = root + "/bin/vipe"
runpy.run_path(sys.argv[0], run_name="__main__")
' "$@"
