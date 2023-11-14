# Description

#!/bin/bash
#SBATCH --time=22:00:00
#SBATCH --ntasks=20
#SBATCH --output=log/tape_embed_%j.output
#SBATCH --error=error/tape_embed_%j.output
#SBATCH --job-name=tape_embed
#SBATCH --partition=gpu_4_h100
#SBATCH --mem=44449mb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu


module unload jupyter/tensorflow/2023-03-24
module load devel/cmake/3.18

module load devel/cuda/11.4
module load devel/cuda/11.4

module unload compiler/intel/2021.4.0
module load compiler/gnu/12.1
module load compiler/gnu/12.1
module unload jupyter/tensorflow/2023-10-10

source /home/kit/aifb/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate base
# conda activate EAsF 
conda activate ss


cd /pfs/work7/workspace/scratch/cc7738-prefeature/subgraph-sketching_chen
cd src

# Experiment 1
# head 1 
python runners/run.py --dataset_name pubmed         --model ELPH  --use_text False --wandb_run_name pubmed_elph      --save_result
python runners/run.py --dataset_name cora           --model ELPH  --use_text False --wandb_run_name cora_elph        --save_result
## not work 
python runners/run.py --dataset_name pubmed         --model BUDDY --use_text False --wandb_run_name pubmed_elph      --save_result
python runners/run.py --dataset_name cora           --model BUDDY --use_text False --wandb_run_name cora_elph        --save_result

# head 2  
python runners/run.py --dataset_name pubmed --model ELPH --use_text True --wandb_run_name pubmed_elph_text --save_result
python runners/run.py --dataset_name cora --model ELPH --use_text True  --wandb_run_name cora_elph_text --save_result 
## not work 

python runners/run.py --dataset_name pubmed --model BUDDY --use_text True --wandb_run_name pubmed_elph_text --save_result
python runners/run.py --dataset_name cora --model BUDDY --use_text True  --wandb_run_name cora_elph_text --save_result


