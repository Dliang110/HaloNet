#!/bin/bash
#SBATCH --job-name=DevHalonetMNIST
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks-per-node=20      # nombre de taches MPI par noeud
#SBATCH --time=05:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=DevHalonetMNIST%j.out          # nom du fichier de sortie
#SBATCH --error=DevHalonetMNIST%j.out     
#SBATCH --account ohv@gpu
#SBATCH --gres=gpu:2# 4
module purge
conda deactivate
module load pytorch-gpu/py3/1.8.1
python train.py

#python test.py
