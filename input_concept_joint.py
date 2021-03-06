import os

import process_funtions as process
import read_files as read

# semantic_type_label = read.textfile2list("data/umls/umls_st.txt")
# semantic_type_label = [item.split('|')[3] for item in semantic_type_label]
# tagger_labels = []
# for label in semantic_type_label:
#     label_new = '_'.join(label.split(' '))
#     tagger_labels.append("B_" + label_new)
#     tagger_labels.append("I_" + label_new)
# tagger_labels.append('O')
# tagger_labels.append('CUI-less')


def get_sg():
    semantic_group_label = read.textfile2list("data/umls/umls_st.txt")
    semantic_type_label = [item.split('|')[1] for item in semantic_group_label]
    sg = []
    for item in semantic_type_label:
        if item not in sg:
            sg.append(item)
    read.save_in_json("data/umls/umls_sg", sg)


# get_sg()


def generate_st_input(file_dir_path, file_path, output_path):
    semantic_type = read.read_from_json(
        "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    # semantic_group = read.read_from_json(
    #     "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    # semantic_group['CUI-less'] = ['CUI_less']
    note_file_name = read.textfile2list(file_dir_path)

    input_new = []

    for note in note_file_name:
        input = read.read_from_json(os.path.join(file_path, note))

        # input = read.read_from_json(file_path)
        tokens_new = []
        tags_new = []
        for tokens, tags in input:
            tokens_new += tokens
            tags_new += tags

        for idx, [token, tag] in enumerate(zip(tokens_new, tags_new)):
            if tag[0] == 'B':
                pos, cid, cui = tag.split('_')[:3]
                sg = '_'.join(
                    process.get_sg_cui(semantic_type, cui).split(' '))
                entity_idx = 1
                entity_text = ['<e>', token]
                while idx + entity_idx <= len(tags_new) - 1 and tags_new[
                        idx + entity_idx][0] not in ['B', 'O']:
                    entity_text.append(tokens_new[idx + entity_idx])
                    entity_idx += 1
                entity_text.append('</e>')

                start = max(0, idx - 10)
                end = min(idx + entity_idx + 10, len(tokens_new) - 1)

                # sentence = tokens_new[start:idx] + entity_text + tokens_new[
                #     idx + entity_idx:end]

                sentence = tokens_new[start:idx] + entity_text + tokens_new[
                    idx + entity_idx:end]

                if "CUI-less" != cui:
                    input_new.append(
                        [sg, cui, " ".join(entity_text),
                        " ".join(sentence)])  ### , " ".join(sentence)

                # input_new.append([" ".join(sentence), cui])
                # input_new.append([
                #     " ".join(entity_text) + " " +
                #     process.get_st_cui(semantic_type, cui), cui
                # ])

    read.save_in_tsv(output_path, input_new)


generate_st_input(
    "data/n2c2/train_dev/train_file_list.txt", "data/n2c2/processed/raw/train",
    "data/n2c2/processed/input_joint/sentence_mention_st_nocuiless/train.tsv")

generate_st_input(
    "data/n2c2/train_dev/dev_file_list.txt", "data/n2c2/processed/raw/dev",
    "data/n2c2/processed/input_joint/sentence_mention_st_nocuiless/dev.tsv")

generate_st_input(
    "data/n2c2/test/test_file_list.txt", "data/n2c2/processed/raw/test",
    "data/n2c2/processed/input_joint/sentence_mention_st_nocuiless/test.tsv")

# generate_st_input("data/n2c2/processed/raw/dev",
#                   "data/n2c2/processed/input_joint_mention/st_eval/dev.tsv")

# generate_st_input("data/n2c2/processed/raw/test",
#                   "data/n2c2/processed/input_joint/st_copy_combine/dev.tsv")


def from_st_to_sg(input_path, output_path):

    semantic_type = read.read_from_json(
        "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    for item in ["train.tsv", "dev.tsv", "test.tsv"]:
        input = read.read_from_tsv(input_path + item)
        input_new = [[
            '_'.join(process.get_sg_cui(semantic_type, cui).split(' ')), cui,
            syn
        ] for [_, cui, syn] in input]
        read.save_in_tsv(output_path + item, input_new)


# from_st_to_sg("data/n2c2/processed/input_joint/mention",
#               "data/n2c2/processed/input_joint/mention_st")

# from_st_to_sg("data/n2c2/processed/input_joint/umls+data/",
#               "data/n2c2/processed/input_joint/umls+data_sg/")


def combine_train_dev():
    umls = read.read_from_tsv("data/n2c2/processed/input_joint/umls/train.tsv")

    train = read.read_from_tsv(
        "data/n2c2/processed/input_joint/mention/train.tsv")

    train_new = umls + train * 60

    read.save_in_tsv("data/n2c2/processed/input_joint/umls+data/train.tsv",
                     train_new)


# combine_train_dev()
