# python train_system_joint.py \
# --model_name_or_path /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/concept_e50_s128/ \
# --data_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/processed/input_joint/st_eval/ \
# --output_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/concept_e50_s128_output/ \
# # --task_name st_joint cn_joint \
# --task_name st_joint \
# # --do_train \
# --do_eval \
# # --do_predict \
# --train_batch_size 16 \
# --num_train_epochs 50 \
# --overwrite_output_dir true \
# --overwrite_cache true \
# --max_seq_length 128 \
# --token true \
# --label_names st_labels

python train_system_joint.py \
--model_name_or_path /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/context_e50_s128_new/checkpoint-8000/ \
--data_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/processed/input_joint/st_eval/ \
--output_dir /home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/context_e50_s128_output/ \
--task_name st_joint \
--do_eval \
--do_predict \
--train_batch_size 16 \
--num_train_epochs 50 \
--overwrite_output_dir true \
--overwrite_cache true \
--max_seq_length 128 \
--token true \
--label_names st_labels