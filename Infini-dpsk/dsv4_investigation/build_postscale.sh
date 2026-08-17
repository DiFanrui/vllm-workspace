#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/InfiniLM-dpv4-test
export INFINI_ROOT=/root/autodl-tmp/dpv4-prefix
export XMAKE_ROOT=y

/root/.local/bin/xmake f -y --use-kv-caching=y
/root/.local/bin/xmake build -y --jobs=4 _infinilm
/root/.local/bin/xmake install -y _infinilm
echo BUILD_POSTSCALE_OK
