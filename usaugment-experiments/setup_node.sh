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

# Copy data and code to compute node ($1 is the dataset name)
tar -xf $project/data/$1.tar.gz -C $SLURM_TMPDIR
rsync -a $project/ultrasound-augmentation $SLURM_TMPDIR --exclude-from=$project/ultrasound-augmentation/.gitignore

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install package and dependencies
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/ultrasound-augmentation/requirements_cc.txt
pip install --no-index -e $SLURM_TMPDIR/ultrasound-augmentation

# Change to working directory
cd $SLURM_TMPDIR
    