#!/bin/bash

cd /scratch/wl2707/research/hilbert_attention/reorder_local_attention

# Submit 20 jobs for partitions 0-19
for partition in {0..19}; do
  sbatch ./scripts/inference.sbatch $partition
done
