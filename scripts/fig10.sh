#!/bin/bash

NPN=8

runtrace() {
    d=$1
    p=$DUMP_PREFIX/$2
    l=$3
    i=$4
    mkdir -p logs/$2/times-$TEST_NAME
    logname=logs/$2/times-$TEST_NAME/log
    touch $logname
    echo D_MODEL $1 TRACE_PATH $2 TRACE_LAYER $3 TRACE_ITER $4 >> $logname
    MASTER_PORT=$(expr $RANDOM % 10000 + 10000) \
    srun --quiet \
        -A priority \
        -p Big \
        -N 2 \
        --exclusive \
        --export=ALL \
        --ntasks-per-node=$NPN \
        --gres=gpu:$NPN \
        --exclusive \
        scripts/exec.sh benchmarks/run_iters.py $d $p $l $i \
        | tee -a $logname | grep Layer
}

runtest() {
    export TEST_NAME=$1

    export FMOE_ENABLE_FUSE=1
    export FMOE_FUSE_GRAN=2
    export FMOE_ENABLE_DYNREP=1

    for dm in 1024 4096
    do
        tp=moe-gpt
        echo Testing $1 on $tp with H=$dm
        rm -f logs/$tp/times-$TEST_NAME/log
        for ti in 500 5500 80500; do
            runtrace $dm $tp 12 $ti
        done

        tp=moe-bert
        echo Testing $1 on $tp with H=$dm
        rm -f logs/$tp/times-$TEST_NAME/log
        for ti in 500 5500 80500; do
            runtrace $dm $tp 24 $ti
        done
    done
}

runds() {
    d=$1
    p=$DUMP_PREFIX/$2
    l=$3
    i=$4
    mkdir -p logs/$2/times-$TEST_NAME
}

run_ds_once() {
    export LOCAL_RANK=0
    export D_MODEL=$1
    export TRACE_PATH=$DUMP_PREFIX/$2
    export DSP_STAGE=$3
    export TEST_NAME=ds-$3
    export TRACE_LAYER=0
    export TRACE_ITER=500
    mkdir -p logs/$2/times-$TEST_NAME/
    logname=logs/$2/times-$TEST_NAME/log
    touch $logname
    echo D_MODEL $1 TRACE_PATH $2 DS Stage $DSP_STAGE >> $logname
    MASTER_PORT=$(expr $RANDOM % 10000 + 10000) \
    srun --quiet \
        -A priority \
        -p Big \
        -N 2 \
        --exclusive \
        --export=ALL \
        --ntasks-per-node=$NPN \
        --gres=gpu:$NPN \
        --exclusive \
        scripts/exec.sh benchmarks/run_ds.py $3 --deepspeed \
            --deepspeed_config benchmarks/deepspeed_config_stage$3.json \
            >> $logname
}

run_ds() {
    for i in {1..3}
    do
        for dm in 1024 4096
        do
            echo "Testing DeepSpeed Stage $i on H=$dm"
            for m in moe-gpt moe-bert
            do
                run_ds_once $dm $m $i
            done
        done
    done
}

