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


def generate_st_input(file_path, output_path):
    semantic_type = read.read_from_json(
        "data/umls/cui_st_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    # semantic_group = read.read_from_json(
    #     "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    # semantic_group['CUI-less'] = ['CUI_less']

    input = read.read_from_json(file_path)
    input_new = []
    for tokens, tags in input:
        if list(set(tags)) != ["O"]:
            for idx, [token, tag] in enumerate(zip(tokens, tags)):
                if tag[0] == 'B':
                    pos, cid, cui = tag.split('_')[:3]
                    st = '_'.join(
                        process.get_st_cui(semantic_type, cui).split(' '))
                    entity_idx = 1
                    entity_text = ['<e>', token]
                    while idx + entity_idx <= len(tags) - 1 and tags[
                            idx + entity_idx][0] not in ['B', 'O']:
                        entity_text.append(tokens[idx + entity_idx])
                        entity_idx += 1
                    entity_text.append('</e>')
                    sentence = tokens[:idx] + entity_text + tokens[idx +
                                                                   entity_idx:]
                    # input_new.append(
                    #     [st, cui, " ".join(entity_text), " ".join(sentence)])

                    input_new.append([st, cui, " ".join(sentence)])

    read.save_in_tsv(output_path, input_new[:1000])


# generate_st_input("data/n2c2/processed/raw/train",
#                   "data/n2c2/processed/input_joint/st_copy_combine/train_sep.tsv")

# generate_st_input("data/n2c2/processed/raw/dev",
#                   "data/n2c2/processed/input_joint/st_copy_combine/dev_sep.tsv")

# generate_st_input("data/n2c2/processed/raw/dev",
#                   "data/n2c2/processed/input_joint_mention/st_eval/dev.tsv")

# generate_st_input("data/n2c2/processed/raw/test",
#                   "data/n2c2/processed/input_joint/st_copy_combine/dev.tsv")


def combine_train_dev():
    train = read.read_from_tsv(
        "data/n2c2/processed/input_joint/st_copy_combine/train_sep.tsv")

    dev = read.read_from_tsv(
        "data/n2c2/processed/input_joint/st_copy_combine/dev_sep.tsv")

    train_new = train + dev

    read.save_in_tsv(
        "data/n2c2/processed/input_joint/st_copy_combine/train.tsv", train_new)


# combine_train_dev()
