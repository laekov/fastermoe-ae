#!/bin/bash

export LOG_PREFIX=ChaosFlow-AE

aelog() {
    echo [$LOG_PREFIX $(date)] $@
}

mkdir -p logs
mkdir -p results

export DUMP_PREFIX=/mnt/zoltan/laekov/dump

reinstall() {
    aelog Uninstalling previous fastmoe
    pip uninstall -y fastmoe 2>&1 >logs/install.log

    aelog Installing $1
    pushd $1
    USE_NCCL=1 TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0" USE_NCCL=1 python3 setup.py install --user 2>&1 >../logs/install.log
    popd 
}

fig9() {
    aelog Running breakdown prediction for fig 9a
    scripts/fig9a.sh
    aelog Running forward and backward prediction for fig 9b
    scripts/fig9b.sh
    aelog Plotting fig 9
    python3 plotting/fig9.py
}

reinstall fastmoe
fig9
