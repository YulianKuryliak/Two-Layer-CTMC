#!/bin/bash
cd ~ || exit 2
mkdir -p texmf
base="https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2023/tlnet-final/archive"
echo "network test:"
curl -sSL -o /tmp/luaotfload.tar.xz -w "http=%{http_code} size=%{size_download}\n" "$base/luaotfload.tar.xz"
echo "file:"
ls -la /tmp/luaotfload.tar.xz
file /tmp/luaotfload.tar.xz
echo "extract:"
tar xf /tmp/luaotfload.tar.xz -C ~/texmf && echo "TAR_OK" || echo "TAR_FAIL"
mktexlsr ~/texmf >/dev/null 2>&1
echo "find:"
find ~/texmf -name 'luaotfload-main.lua' | head -1
