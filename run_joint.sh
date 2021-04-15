# python train_system_joint.py \
# --model_name_or_path /home/dongfangxu/Projects/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
# --data_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/processed/input_joint/st_copy/ \
# --output_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/joint_training/ \
# --task_name st_joint cn_joint \
# --do_train \
# --do_eval \
# --do_predict \
# --train_batch_size 16 \
# --num_train_epochs 50 \
# --overwrite_output_dir true \
# --overwrite_cache true \
# --max_seq_length 128 \
# --token true \
# --label_names st_labels concept_labels

python3.7 train_system_joint.py \
--model_name_or_path /home/dongfangxu/Projects/models/0_Transformer \
--data_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/processed/input_joint/st_copy_combine/ \
--output_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/joint_training/ \
--task_name st_joint \
--do_train \
--do_eval \
--train_batch_size 16 \
--num_train_epochs 10 \
--overwrite_output_dir true \
--overwrite_cache true \
--max_seq_length 128 \
--token true \
--label_names st_labels \
--pad_to_max_length true