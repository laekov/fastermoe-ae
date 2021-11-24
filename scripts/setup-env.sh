#!/bin/bash
source /opt/spack/share/spack/setup-env.sh
spack load nccl@2.10.3
NCCL_PATH=$(echo $CMAKE_PREFIX_PATH | tr "=" "\n" | tr ":" "\n" | grep nccl)
NCCL_INCLUDE_PATH=$NCCL_PATH/include
NCCL_LIB_PATH=$NCCL_PATH/lib
export C_INCLUDE_PATH=$NCCL_INCLUDE_PATH:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$NCCL_INCLUDE_PATH:$CXX_INCLUDE_PATH
export LIBRARY_PATH=$NCCL_LIB_PATH:$LIBRARY_PATH
