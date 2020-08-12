#!/usr/bin/env bash

PATH='/opt/conda/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

if [ ! -d "imgs" ]; then
    # get celeba data
    DEBIAN_FRONTEND=noninteractive apt-get -yq update
    DEBIAN_FRONTEND=noninteractive apt-get -yq install wget unzip git

    echo "getting data"
    wget --quiet --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g9tAydWEN0PW-BYxdhNuGG9vXrdNzffA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g9tAydWEN0PW-BYxdhNuGG9vXrdNzffA" -O img_align_celeba.zip && rm -rf /tmp/cookies.txt
    unzip -q img_align_celeba.zip -d imgs

    # install apex
    git clone https://github.com/NVIDIA/apex.git
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

    # get FID calculation library
    git clone https://github.com/mseitzer/pytorch-fid.git
    mv pytorch-fid stylegan2-pytorch/stylegan2_pytorch/pytorch_fid
    touch stylegan2-pytorch/stylegan2_pytorch/pytorch_fid/__init__.py
    sed -i 's/from pytorch_fid.inception/from .inception/' stylegan2-pytorch/stylegan2_pytorch/pytorch_fid/pytorch_fid/fid_score.py

    # get some more python libraries
    pip install matplotlib scipy
fi

pip install stylegan2-pytorch/
chmod u+x stylegan2-pytorch/bin/stylegan2_pytorch