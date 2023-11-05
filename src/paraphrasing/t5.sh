#!/bin/bash
#SBATCH -A aparna
#SBATCH -c 2
#SBATCH -w gnode017
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python  t5_text_simplification_controlled.py
