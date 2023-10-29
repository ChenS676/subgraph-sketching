# Description
module unload jupyter/tensorflow/2023-10-10 
cd /pfs/work7/workspace/scratch/cc7738-prefeature/subgraph-sketching_chen/src
# Experiment 1
# head 1 
python runners/run.py  --use_text False  --dataset_name pubmed     --model ELPH   --wandb_run_name pubmed_elph  --wandb_group reconstruct --wandb_tags baseline
python runners/run.py  --use_text False  --dataset_name pubmed     --model BUDDY  --wandb_run_name pubmed_buddy

python runners/run.py  --use_text False  --dataset_name cora       --model ELPH   --wandb_run_name cora_elph
python runners/run.py  --use_text False  --dataset_name cora       --model BUDDY  --wandb_run_name cora_buddy

## not work 
python runners/run.py  --use_text False  --dataset_name ogbn-arxiv --model ELPH   --wandb_run_name ogbn-arxiv_elph
python runners/run.py  --use_text False  --dataset_name ogbn-arxiv --model BUDDY  --wandb_run_name ogbn-arxiv_buddy

# head 2 
python runners/run.py  --use_text True   --dataset_name pubmed      --model ELPH   --wandb_run_name pubmed_elph_text
python runners/run.py  --use_text True   --dataset_name pubmed      --model BUDDY  --wandb_run_name pubmed_buddy_text

# python runners/run.py  --use_text True   --dataset_name cora        --model ELPH   --wandb_run_name cora_elph_text
# python runners/run.py  --use_text True   --dataset_name cora        --model BUDDY  --wandb_run_name cora_buddy_text

## not work 
# python runners/run.py  --use_text True   --dataset_name ogbn-arxiv  --model ELPH   --wandb_run_name ogbn-arxiv_elph_text
# python runners/run.py  --use_text True   --dataset_name ogbn-arxiv  --model BUDDY   --wandb_run_name ogbn-arxiv_buddy_text
