#!/bin/bash

mkdir -p logs
mkdir -p results

export DUMP_PREFIX=/mnt/zoltan/laekov/dump

scripts/fig9a.sh
scripts/fig9b.sh
