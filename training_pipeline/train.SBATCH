#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=Flux
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=tandon_a100_1,tandon_a100_2,stake_a100_1,stake_a100_2
#SBATCH --account=pr_133_tandon_priority
#SBATCH --mem=120G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=yx2432

module purge

TARGET_PATH=$(dirname "$(realpath "$0")")

singularity exec --nv --bind /scratch/$USER \
	    --overlay /scratch/yx2432/singularity/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; cd $TARGET_PATH; bash train.sh"

