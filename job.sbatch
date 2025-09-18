#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=2:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1


current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 75f25bb15ce90a3068dc9b029329aa7934a029d8"
cd $TMPDIR/code
git checkout 75f25bb15ce90a3068dc9b029329aa7934a029d8


echo "Setting up the virtual environment"
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate


echo "check config file exists"
if [ ! -f "/usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_34/deep-learning-project/configs/tmp96iyah9o-config.yml" ]; then
    echo "Config file /usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_34/deep-learning-project/configs/tmp96iyah9o-config.yml not found. Aborting."
    exit 1
else
    echo "Config file exists"
fi


# Install the library
python -m pip install .
python -m pip install --upgrade rasterio attr
python -m pip install opencv-python
python -m pip install albumentations


echo "Setting up your package"
python setup.py install


echo "Training"

python -m torchtmpl.main /usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_34/deep-learning-project/configs/tmp96iyah9o-config.yml test
echo "Done"
if [[ $? != 0 ]]; then
    echo "Error: went to the if loop"
    exit -1
fi

