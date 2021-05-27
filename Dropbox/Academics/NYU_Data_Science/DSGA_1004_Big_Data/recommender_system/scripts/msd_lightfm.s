#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=60:00:00
#SBATCH --mem=64GB
#SBATCH --chdir=/home/jz3786/1004/final-project-personupperson/
#SBATCH --job-name=msd30_0
#SBATCH --output=log/msd_lightfm_rank30_alpha0.out

source ~/.bashrc
conda activate py36

# python extension_single/msd_lightfm.py --rank 100 --valid_duration 5 --alpha 0.1 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank100_alpha01.pkl
# python extension_single/msd_lightfm.py --rank 30 --valid_duration 5 --alpha 0.1 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank30_alpha01.pkl
# python extension_single/msd_lightfm.py --rank 10 --valid_duration 5 --alpha 0.1 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank10_alpha01.pkl

# python extension_single/msd_lightfm.py --rank 100 --valid_duration 5 --alpha 0.01 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank30_alpha001.pkl
# python extension_single/msd_lightfm.py --rank 30 --valid_duration 5 --alpha 0.01 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank30_alpha001.pkl
# python extension_single/msd_lightfm.py --rank 10 --valid_duration 5 --alpha 0.01 --num_threads 40 --epochs 20 --earlystop_patient 2 --outfile out/msd_lightfm_rank30_alpha001.pkl

python extension_single/msd_lightfm.py --rank 100 --valid_duration 5 --num_threads 20 --epochs 40 --earlystop_patient 2 --outfile out/msd_lightfm_rank100_alpha0.pkl
python extension_single/msd_lightfm.py --rank 30 --valid_duration 5 --num_threads 20 --epochs 40 --earlystop_patient 2 --outfile out/msd_lightfm_rank30_alpha0.pkl
python extension_single/msd_lightfm.py --rank 10 --valid_duration 5 --num_threads 20 --epochs 40 --earlystop_patient 2 --outfile out/msd_lightfm_rank10_alpha0.pkl









