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
    aelog Running breakdown prediction of fig 9a
    scripts/fig9a.sh
    aelog Running forward and backward prediction of fig 9b
    scripts/fig9b.sh
    aelog Plotting fig 9
    python3 plotting/fig9.py
}

fig10() {
    source scripts/fig10.sh
    # aelog Running baseline per-iteration performance
    # runtest fastmoe
    # aelog Running ZerO baselines
    # run_ds
    reinstall chaosflow
    aelog Running ChaosFlow
    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=1
    runtest chaosflow
}

fig13() {
    source scripts/fig10.sh
    aelog Running shadowing
    export FMOE_ENABLE_FUSE=0
    export FMOE_FUSE_GRAN=0
    export FMOE_ENABLE_DYNREP=1
    runtest dynrep
    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=0
    runtest smartsch
}

# reinstall fastmoe
# fig9
# fig10
fig13
