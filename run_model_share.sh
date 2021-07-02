#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=12:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/0.1dropout_0.85_fixed_concept_umls+train+dev%j.txt    # Name of the output file
#SBATCH --error=log/0.1dropout_0.85_fixed_concept_umls+train+dev%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity


OUTPUT_DIR=/temp_work/ch223150/outputs/share/0.1dropout_0.85_fixed_concept_umls+train+dev_umls+train_triplet_ontology+train_all_e20_b400_seq16_5e5_sc45_m0.35

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 train_system_conceptnorm.py \
        --model_name_or_path /temp_work/ch223150/outputs/share/umls+train_triplet_ontology+train_all_e20_b400_seq16_5e5_sc45_m0.35/checkpoint-9576/bert/ \
        --data_dir /home/ch223150/projects/Concept_Norm/data/share/umls+data/ \
        --output_dir $OUTPUT_DIR \
        --task_name concept_normalization \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size 128 \
        --num_train_epochs 20 \
        --overwrite_output_dir true \
        --overwrite_cache true \
        --max_seq_length 16 \
        --token true \
        --label_names concept_labels \
        --pad_to_max_length true \
        --learning_rate 5e-5 \
        --margin 0.45 \
        --scale 30 \
        --concept_embeddings_pre true

