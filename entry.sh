#!/usr/bin/env bash

python -m visdom.server & > visdom_log

/bin/sleep 20

echo "training GAN"

python ./src/3dgan_mit_biasfree.py 0 0
