#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=2:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/exp_%j.txt    # Name of the output file
#SBATCH --error=log/exp_%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity

# scales='5 10 15 20 25 30 35 40 45'
# for scale in $scales
# do

# DATA_DIR=/home/ch223150/projects/Concept_Norm/data/n2c2/joint_input/i2b2_2010/context_full/
# OUTPUT_DIR = /temp_work/ch223150/outputs/share/0.65_fixed_concept_umls+train+dev_umls+data_umls+data_c4255_e20_b400_seq16_5e5_sc45_m0.35/checkpoint-39891/



singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 train_system_joint.py \
--model_name_or_path /temp_work/ch223150/outputs/share/0.65_fixed_concept_umls+train+dev_umls+data_umls+data_c4255_e20_b400_seq16_5e5_sc45_m0.35/checkpoint-39891/ \
--data_dir /home/ch223150/projects/Concept_Norm/data/n2c2/joint_input/i2b2_2010/context_full/ \
--output_dir /temp_work/ch223150/outputs/share/0.65_fixed_concept_umls+train+dev_umls+data_umls+data_c4255_e20_b400_seq16_5e5_sc45_m0.35/checkpoint-39891/ \
--task_name st_joint \
--do_predict \
--overwrite_output_dir true \
--overwrite_cache true \
--max_seq_length 16 \
--token true \
--label_names st_labels \
--pad_to_max_length true
