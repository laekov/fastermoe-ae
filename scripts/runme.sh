#!/bin/bash

export LOG_PREFIX=ChaosFlow-AE

# Update 2 paths below on your own cluster
export DUMP_PREFIX=/mnt/zoltan/laekov/dump
export DATA_PREFIX=/home/ppopp_ae/dataset
# Number of GPUs per node
export NPN=8

aelog() {
    echo [$LOG_PREFIX $(date)] $@
}

mkdir -p logs
mkdir -p results

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
    aelog Running baseline per-iteration performance
    runtest fastmoe
    aelog Running ZerO baselines
    run_ds
    reinstall chaosflow
    aelog Running ChaosFlow
    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=1
    runtest chaosflow
    aelog Plotting fig 10
    python3 plotting/fig10.py
}

fig11() {
    aelog Plotting fig 11
    python3 plotting/fig11.py
}

fig12() {
    source scripts/fig12.sh
    aelog Preparing prediction of fig 12
    parse_cache
    sim_cache
    aelog Plotting fig 12
    python3 plotting/fig12.py
}

fig13() {
    source scripts/fig10.sh
    aelog Running shadowing
    export FMOE_ENABLE_FUSE=0
    export FMOE_FUSE_GRAN=0
    export FMOE_ENABLE_DYNREP=1
    runtest dynrep

    aelog Running smart schedule
    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=0
    runtest smartsch

    aelog Plotting fig 13
    python3 plotting/fig13.py
}

tab3() {
    reinstall fastmoe
    source scripts/tab3.sh

    aelog Running FastMoE, training 500 iterations each
    BAL_STG=naive pretrain fastmoe

    aelog Running GShard
    BAL_STG=gshard pretrain gshard

    reinstall chaosflow

    aelog Running BASE Layers
    BAL_STG=baseorig pretrain baselayers

    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=1
    aelog Running ChaosFlow with only shadow and smart scheduling
    BAL_STG=naive pretrain chaosflow

    aelog Running ChaosFlow with topo-gate
    BAL_STG=hir pretrain topogate
    gentab3 | tee results/table3.txt
}

reinstall fastmoe
fig9
fig10
fig11
fig12
fig13
tab3
