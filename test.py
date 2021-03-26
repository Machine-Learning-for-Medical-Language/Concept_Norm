import transformers

# model = AutoModel.from_pretrained(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
import read_files as read

# tokenizer = AutoTokenizer.from_pretrained("/home/dongfangxu/Projects/models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
# input = [[
#     'B-ORG O B-ORG O O O B-PER O O',
#     'EU rejects German call to boycott British lamb .'
# ]] * 1000

# print(input[0][0])
# read.save_in_tsv("data/ner_test/train.tsv", input)
# read.save_in_tsv("data/ner_test/dev.tsv", input)
# read.save_in_tsv("data/ner_test/test.tsv", input)

# # print(input)
# Labels = [
#     'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC',
#     'I-MISC'
# ]

# from datasets import load_dataset, load_metric

# datasets = load_dataset("conll2003")

# from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

semantic_type_label = read.textfile2list("data/umls/umls_st.txt")

semantic_type_label = [item.split('|')[3] for item in semantic_type_label]
tagger_labels = []
for label in semantic_type_label:
    label_new = '_'.join(label.split(' '))
    tagger_labels.append("B-" + label_new)
    tagger_labels.append("I-" + label_new)
tagger_labels.append('O')
tagger_labels.append('B-CUIless')
tagger_labels.append('I-CUIless')
read.save_in_json("data/umls/umls_st_tagging", tagger_labels)
print(tagger_labels)
