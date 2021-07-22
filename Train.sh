#!/bin/bash
#SBATCH --job-name=DevHalonet
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=00:10:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=DevHalonet%j.out          # nom du fichier de sortie
#SBATCH --error=DevHalonet%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:2# 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python VH_ADL.py
#python test.py
