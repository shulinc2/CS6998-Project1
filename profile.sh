#!/bin/bash

MODEL=$1
GPU=$2
NCU_PATH="/share/apps/cuda/11.1.74/bin/ncu"
NSYS_PATH="/usr/local/cuda-11.0/bin/nsys"

CMD="python3 main.py -a $MODEL  --pretrained --gpu 0  --epoch 1 /imagenette2-160"
METRICS="dram__sectors.sum,dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"

echo "********** NCU PROFILING STARTS **********"

$NCU_PATH  --set roofline --metrics $METRICS -o output --target-processes all $CMD

