import read_files as read

semantic_type = read.read_from_json(
    "data/umls/cui_st_term_snomed_rxnorm_dict_all")
semantic_type['CUI-less'] = ['CUI_less']

# semantic_type_label = read.textfile2list("data/umls/umls_st.txt")
# semantic_type_label = [item.split('|')[3] for item in semantic_type_label]
# tagger_labels = []
# for label in semantic_type_label:
#     label_new = '_'.join(label.split(' '))
#     tagger_labels.append("B_" + label_new)
#     tagger_labels.append("I_" + label_new)
# tagger_labels.append('O')
# tagger_labels.append('CUI-less')


def get_st_cui(cui):
    cui_st_list = semantic_type[cui]
    # elif "Pharmacologic Substance" in semantic_type[concepts[1]]:
    # #     cui_st = ["Pharmacologic Substance"]
    if len(cui_st_list) > 1:
        if "Pharmacologic Substance" in cui_st_list:
            cui_st = ["Pharmacologic Substance"]
        elif "Antibiotic" in cui_st_list:
            cui_st = ["Antibiotic"]
        elif "Biologically Active Substance" in cui_st_list:
            cui_st = ["Biologically Active Substance"]
        elif "Manufactured Object" in cui_st_list:
            cui_st = ["Manufactured Object"]
        else:
            cui_st = cui_st_list[:1]
        return cui_st[0]
    else:
        return cui_st_list[0]


def generate_st_input(file_path, output_path):
    input = read.read_from_json(file_path)
    input_new = []
    for tokens, tags in input:
        tag_st = []
        for tag in tags:
            if tag == 'O':
                tag_st.append('O')
            else:
                pos, cid, cui = tag.split('_')[:3]
                st = '_'.join(get_st_cui(cui).split(' '))
                tag_st.append(pos + '-' + st)
        tokens = ' '.join(tokens)
        tag_sts = ' '.join(tag_st)

        if len(tokens.split(" ")) != len(tag_sts.split(" ")):
            ValueError("Number of tokens != Number of spans")
        input_new.append([tag_sts, tokens])
    read.save_in_tsv(output_path, input_new)


generate_st_input("data/n2c2/processed/raw/train",
                  "data/n2c2/processed/input/train.tsv")

generate_st_input("data/n2c2/processed/raw/dev",
                  "data/n2c2/processed/input/dev.tsv")

generate_st_input("data/n2c2/processed/raw/test",
                  "data/n2c2/processed/input/test.tsv")
