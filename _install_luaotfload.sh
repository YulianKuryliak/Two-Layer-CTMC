#!/bin/bash
set -e
mkdir -p ~/texmf
base=https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2023/tlnet-final/archive
wget -q "$base/luaotfload.tar.xz" -O /tmp/luaotfload.tar.xz
tar xf /tmp/luaotfload.tar.xz -C ~/texmf
mktexlsr ~/texmf 2>&1 | tail -2
echo "--- find ---"
find ~/texmf -name 'luaotfload-main.lua' | head -1
