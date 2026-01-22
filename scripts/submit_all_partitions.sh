#!/bin/bash

# Submit 20 jobs for partitions 0-19
for partition in {0..19}; do
  sbatch /home/wl2707/Projects/reorder_local_attention/inference.sbatch $partition
done
