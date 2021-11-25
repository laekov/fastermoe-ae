#!/bin/bash

instapex() {
    pushd apex
    python3 setup.py install --user --cpp_ext --cuda_ext 2>&1 >../logs/install.log
    popd
}

pretrain() {
    export TASK_NAME=train_$1
    pushd Megatron-LM

    echo Training with strategy $BAL_STG for 500 iterations
    echo "    use this command to see the training process"
    echo "        tail -f logs/$TASK_NAME.log"

    MASTER_PORT=$(expr $RANDOM % 10000 + 10000) \
    srun --quiet \
        -A priority \
        -p Big \
        -N 2 \
        --exclusive \
        --export=ALL \
        --ntasks-per-node=1 \
        --gres=gpu:$NPN \
        --exclusive \
        -o ../logs/$TASK_NAME.log \
        examples/pretrain_gpt_distributed.sh 
    popd
}

gentab3() {
    for f in fastmoe gshard baselayers chaosflow topogate
    do
        avgt=$(cat logs/train_$f.log | tr "|" "\n" | grep "per iteration"  | awk '{print $NF}' | python3 -c "import numpy; import sys; print('{:.3f}'.format(numpy.mean(list(map(float, sys.stdin.read().split())))))")
        echo $f : $avgt ms / iteration
    done
}
