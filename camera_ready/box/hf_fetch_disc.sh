#!/bin/bash
set +x
export HISTFILE=/dev/null
source /etc/network_turbo >/dev/null 2>&1
export HF_HOME=/root/autodl-tmp/hf-upload-home
nice -n 19 ionice -c3 /root/autodl-tmp/envs/hfdl/bin/python /root/autodl-tmp/hf_fetch2.py /root/autodl-tmp/a40/fetch_manifest_discovery.json 8 >> /root/autodl-tmp/logs/hf_fetch_discovery.log 2>&1
echo "exit=$?" >> /root/autodl-tmp/logs/hf_fetch_discovery.log
unset HF_TOKEN http_proxy https_proxy
