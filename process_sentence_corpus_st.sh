python3 process_sentence_corpus_st.py \
--model /home/dongfangxu/Projects/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/ \
--model_type bert \
--sentences /home/dongfangxu/Projects/Concept_Norm/data/n2c2/triplet_network/con_norm_alllow/ontology_synonyms.tsv \
--output /home/dongfangxu/Projects/Concept_Norm/data/pubmed1/ontology+train+dev_syn_embeddings

python3 average_embeddings.py \
--syn_path /home/dongfangxu/Projects/Concept_Norm/data/pubmed1/ontology+train+dev_syn_embeddings.npy \
--cui_path /home/dongfangxu/Projects/Concept_Norm/data/n2c2/triplet_network/st_subpool/ontology_cui \
--cui_idx_path /home/dongfangxu/Projects/Concept_Norm/data/n2c2/triplet_network/con_norm_alllow/ontology_concept_synonyms_idx \
--file_name /home/dongfangxu/Projects/Concept_Norm/data/pubmed1/ontology+train+dev_con_embeddings

