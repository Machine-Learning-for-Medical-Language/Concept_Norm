#!/bin/bash
#SBATCH --partition=chip-gpu             # queue to be used
#SBATCH --account=chip
#SBATCH --time=1:00:00             # Running time (in hours-minutes-seconds)
#SBATCH --job-name=conorm             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL      # send and email when the job begins, ends or fails
#SBATCH --mail-user=dongfang.xu@childrens.harvard.edu      # Email address to send the job status
#SBATCH --output=log/exp_margin%j.txt    # Name of the output file
#SBATCH --error=log/exp_margin%j.err
#SBATCH --nodes=1               # Number of gpu nodes
#SBATCH --gres=gpu:Titan_RTX:1                # Number of gpu devices on one gpu node


pwd; hostname; date

module load singularity

OUTPUT_DIR=/temp_work/ch223150/outputs/joint_model/new_joint_st_cn_ontology+train_all_e10_b400_seq16_1e4_sc45_m0.35/checkpoint-19947/
OUTPUT_DIR_BERT=$OUTPUT_DIR/bert

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 extract_bert.py \
--model_path $OUTPUT_DIR \
--save_path $OUTPUT_DIR_BERT

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 process_sentence_corpus_st.py \
--model $OUTPUT_DIR_BERT \
--model_type bert \
--sentences /home/ch223150/projects/Concept_Norm/data/n2c2/triplet_network/con_norm_alllow/ontology_synonyms.tsv \
--output $OUTPUT_DIR_BERT/ontology+train+dev_syn_embeddings

singularity exec -B $TEMP_WORK --nv /temp_work/ch223150/image/hpc-ml_centos7-python3.7-transformers4.4.1.sif  python3.7 average_embeddings.py \
--syn_path $OUTPUT_DIR_BERT/ontology+train+dev_syn_embeddings.npy \
--cui_path /home/ch223150/projects/Concept_Norm/data/n2c2/triplet_network/st_subpool/ontology_cui \
--cui_idx_path /home/ch223150/projects/Concept_Norm/data/n2c2/triplet_network/con_norm_alllow/ontology_concept_synonyms_idx \
--file_name $OUTPUT_DIR_BERT/ontology+train+dev_con_embeddings

