#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=12:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/1n2c2_i2b2%j.txt    # Name of the output file
#SBATCH --error=log/1n2c2_i2b2%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity


OUTPUT_DIR=/temp_work/ch223150/outputs/n2c2/1n2c2_st_n2c2_i2b2

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 train_system_semantictype.py \
        --model_name_or_path /home/ch223150/projects/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/ \
        --data_dir /home/ch223150/projects/Concept_Norm/data/n2c2/joint_input/n2c2_i2b2/ \
        --output_dir $OUTPUT_DIR \
        --task_name semantic_type \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size 64 \
        --num_train_epochs 10 \
        --overwrite_output_dir true \
        --overwrite_cache true \
        --max_seq_length 128 \
        --token true \
        --label_names concept_labels \
        --pad_to_max_length true \
        --learning_rate 5e-5

