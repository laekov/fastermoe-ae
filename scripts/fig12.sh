#!/bin/bash

genprfxs() {
    for ti in 500 5500 80500
    do
        for l in {0..11}
        do
            echo $DUMP_PREFIX/moe-gpt $l $ti
        done
        for l in {0..23}
        do
            echo $DUMP_PREFIX/moe-bert $l $ti
        done
    done
}

parse_cache() {
    mkdir -p cache
    genprfxs | xargs -n 3 -P 164 python3 scripts/trace2lec.py >logs/dump.log
}

sim_cache() {
    for d_model in 1024 4096
    do
        export D_MODEL=$d_model
        export LOG_PREFIX=logs/sims/$D_MODEL
        mkdir -p $LOG_PREFIX
        ls cache/* | xargs -P 70 -n 1 python3 benchmarks/flexible_sim.py >logs/sim.log
    done
}

sim_cache
