#!/bin/bash

#SBATCH --job-name=adv_classifier_test_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=20000M
#SBATCH --account=math027744
#SBATCH --error=colour_patch_error_gpu.txt
#SBATCH --output=colour_patch_output_gpu.txt

source activate

conda activate tf-env

cd "${SLURM_SUBMIT_DIR}"
echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

python Adversarial-Classifier-Coloured-Patch-Detection.py