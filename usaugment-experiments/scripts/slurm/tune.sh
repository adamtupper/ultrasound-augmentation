#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:v100:1
#SBATCH --array 1-100%2 
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL

# Print Job info
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Purge modules
module purge

# Load modules
module load python/3.11 cuda cudnn rust httpproxy opencv

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME=$project
export NO_ALBUMENTATIONS_UPDATE=1
export HYDRA_FULL_ERROR=1

# Check for dataset
if [ -z "1" ]; then
    echo "No dataset specified"
    exit 1
fi

# Check for task
if [ -z "2" ]; then
    echo "No task specified"
    exit 1
fi

# Copy data and code to compute node
tar -xf $project/data/$1.tar.gz -C $SLURM_TMPDIR
rsync -a $project/usaugment-experiments $SLURM_TMPDIR --exclude-from=$project/usaugment-experiments/.gitignore

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install package and dependencies
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/usaugment-experiments/requirements_cc.txt
pip install --no-index -e $SLURM_TMPDIR/usaugment-experiments

# Change to working directory
cd $SLURM_TMPDIR

# Create output directory
mkdir -p $scratch/$1/$2

# Run trial
python usaugment-experiments/src/usaugment/tune.py \
    output_dir=$scratch/$1/$2 \
    data_dir=$SLURM_TMPDIR \
    task=$2
    