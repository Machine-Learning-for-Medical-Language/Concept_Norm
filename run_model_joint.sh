#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=10:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/exp_margin%j.txt    # Name of the output file
#SBATCH --error=log/exp_margin%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity


OUTPUT_DIR=/temp_work/ch223150/outputs/joint_model/1new_joint_st_cn_ontology+train_all_e10_b400_seq16_1e4_sc45_m0.35

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif python3.7 train_system_joint.py \
        --model_name_or_path /temp_work/ch223150/outputs/joint_model/new_joint_st_cn_ontology+train_all_e10_b400_seq16_1e4_sc45_m0.35/checkpoint-19947/bert/ \
        --data_dir /home/ch223150/projects/Concept_Norm/data/n2c2/joint_input/umls+data/ \
        --output_dir $OUTPUT_DIR \
        --task_name st_joint cn_joint \
        --do_train \
        --do_eval \
        --train_batch_size 256 \
        --num_train_epochs 5 \
        --overwrite_output_dir true \
        --overwrite_cache false \
        --max_seq_length 16 \
        --token true \
        --label_names st_labels concept_labels \
        --pad_to_max_length true \
        --learning_rate 5e-5 \
        --margin 0.35 \
        --scale 30 \
        --evals_per_epoch 1
