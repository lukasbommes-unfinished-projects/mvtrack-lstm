#!/bin/sh
set -e

conda install -y av -c conda-forge  # needed to fix ffmpeg error during torchvision build
export FORCE_CUDA=1
cd /opt/pytorch
rm -rf vision
git config --global user.email "Lukas.Bommes@gmx.de"
git config --global user.name "LukasBommes"
git clone https://github.com/LukasBommes/vision.git
cd vision
pip install -v .
cd /workspace

# downgrade pillow to version before 7.0.0 to prevent incompatibility with torchvision
/opt/conda/bin/conda install -y 'pillow<7'

exec "$@"
