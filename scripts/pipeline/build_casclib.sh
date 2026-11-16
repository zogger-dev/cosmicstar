#!/usr/bin/env bash
# Builds CascLib (https://github.com/ladislav-zezula/CascLib, MIT) into lib/libcasc.so for
# envs/casc.py, which reads the game's data files out of the local CASC storage by name
# (StarCraft II's MNDX root). Plain g++ / gcc, no cmake: the C files (jenkins hash, bundled
# zlib) are compiled as C, the rest as C++17, then linked into one shared library.
#
#   scripts/build_casclib.sh [path/to/CascLib checkout]   (cloned to ~/src/CascLib if absent)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HOME/src/CascLib}"
if [ ! -d "$SRC/src" ]; then
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 https://github.com/ladislav-zezula/CascLib.git "$SRC"
fi
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cd "$SRC"
for f in src/jenkins/lookup3.c src/zlib/adler32.c src/zlib/crc32.c src/zlib/inffast.c src/zlib/inflate.c src/zlib/inftrees.c src/zlib/zutil.c; do
    gcc -O2 -fPIC -w -c "$f" -o "$BUILD/$(basename "$f").o"
done
# exactly CMakeLists.txt's SRC_FILES: src/overwatch/cmf-key.cpp is #included by cmf.cpp, not a unit of its own
CPP="src/common/Common.cpp src/common/Directory.cpp src/common/Csv.cpp src/common/FileStream.cpp src/common/FileTree.cpp
     src/common/ListFile.cpp src/common/Mime.cpp src/common/RootHandler.cpp src/common/Sockets.cpp
     src/hashes/md5.cpp src/hashes/sha1.cpp src/overwatch/apm.cpp src/overwatch/cmf.cpp src/overwatch/aes.cpp
     src/CascDecompress.cpp src/CascDecrypt.cpp src/CascDumpData.cpp src/CascFiles.cpp src/CascFindFile.cpp
     src/CascIndexFiles.cpp src/CascOpenFile.cpp src/CascOpenStorage.cpp src/CascReadFile.cpp
     src/CascRootFile_Diablo3.cpp src/CascRootFile_Install.cpp src/CascRootFile_MNDX.cpp src/CascRootFile_Text.cpp
     src/CascRootFile_TVFS.cpp src/CascRootFile_OW.cpp src/CascRootFile_WoW.cpp"
for f in $CPP; do
    g++ -std=c++17 -O2 -fPIC -w -c "$f" -o "$BUILD/$(basename "$f").o"
done
mkdir -p "$REPO/lib"
g++ -shared -o "$REPO/lib/libcasc.so" "$BUILD"/*.o -lpthread
echo "built $REPO/lib/libcasc.so ($(nm -D "$REPO/lib/libcasc.so" | grep -c ' T Casc') Casc* symbols)"
