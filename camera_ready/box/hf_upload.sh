#!/bin/bash
# Upload launcher (round $1). HF_TOKEN arrives through this screen session's environment only.
# Proxy is enabled only inside this shell; the training/evaluation processes never see it.
set +x
export HISTFILE=/dev/null
source /etc/network_turbo >/dev/null 2>&1
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
export HF_HOME=/root/autodl-tmp/hf-upload-home   # separate, token-free HF home (nothing is logged in)
mkdir -p /root/autodl-tmp/logs "$HF_HOME"
nice -n 19 ionice -c3 /root/autodl-tmp/envs/hfdl/bin/python /root/autodl-tmp/repos/cr/camera_ready/box/hf_upload.py --round "$1" \
  >> "/root/autodl-tmp/logs/hf_upload_round$1.out" 2>&1
echo "exit=$?" >> "/root/autodl-tmp/logs/hf_upload_round$1.out"
unset HF_TOKEN http_proxy https_proxy
