#!/usr/bin/env bash

python3 -m visdom.server & > visdom_log

/bin/sleep 20

echo "training GAN"

python3 ./src/3dgan_mit_biasfree.py 0 0
