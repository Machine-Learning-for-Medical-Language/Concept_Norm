#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=2:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/exp_1%j.txt    # Name of the output file
#SBATCH --error=log/exp_1%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 train_system_joint.py \
--model_name_or_path /home/ch223150/projects/models/0_Transformer \
--data_dir /home/ch223150/projects/Concept_Norm/data/n2c2/joint_input/st_all/ \
--output_dir /temp_work/ch223150/outputs/joint_model/e20_b16_s128_5e5 \
--task_name st_joint \
--do_train \
--do_eval \
--do_predict \
--train_batch_size 16 \
--num_train_epochs 20 \
--overwrite_output_dir true \
--overwrite_cache true \
--max_seq_length 128 \
--token true \
--label_names st_labels \
--pad_to_max_length true
