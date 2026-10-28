#!/bin/bash
base="https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2023/tlnet-final/archive"
cd ~/texmf || exit 2
for p in lua-uni-algos lua-alt-getopt; do
  echo "=== $p ==="
  curl -k -sSL -o /tmp/$p.tar.xz -w "http=%{http_code} size=%{size_download}\n" "$base/$p.tar.xz"
  tar xf /tmp/$p.tar.xz -C ~/texmf && echo "TAR_OK $p" || echo "TAR_FAIL $p"
done
cp -r ~/texmf/texmf-dist/* ~/texmf/ 2>/dev/null
rm -rf ~/texmf/texmf-dist
mktexlsr ~/texmf >/dev/null 2>&1
echo "lua-uni-case: $(kpsewhich lua-uni-case.lua)"
