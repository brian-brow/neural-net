#!/bin/sh
# Download and verify the MNIST dataset into data/.
#
# data/ is gitignored, so this script is how the dataset is reproduced on a
# fresh clone. yann.lecun.com/exdb/mnist no longer serves these files; this is
# the mirror torchvision uses.
set -eu

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
DEST="$(dirname "$0")/../data"

FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

mkdir -p "$DEST"
cd "$DEST"

for name in $FILES; do
  if [ -f "$name" ]; then
    echo "$name already present, skipping"
    continue
  fi

  echo "downloading $name.gz"
  curl -sSL --retry 3 --max-time 300 -o "$name.gz" "$BASE/$name.gz"
done

# Checksums are of the compressed files, so verify before unpacking.
for name in $FILES; do
  [ -f "$name.gz" ] || continue

  case "$name" in
    train-images-*) sum=f68b3c2dcbeaaa9fbdd348bbdeb94873 ;;
    train-labels-*) sum=d53e105ee54ea40749a09fcbcd1e9432 ;;
    t10k-images-*)  sum=9fb629c4189551a2d022fa330f9573f3 ;;
    t10k-labels-*)  sum=ec29112dd5afa0611ce80d1b7f02629c ;;
  esac

  echo "$sum  $name.gz" | md5sum -c -
  gunzip -f "$name.gz"
done

echo "MNIST ready in $DEST"
ls -la
