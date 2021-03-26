from collections import Counter

import process_funtions as process
import read_files as read


def analyze_st_freq(file_dir_path, notedir):
    semantic_type = read.read_from_json(
        "data/umls/cui_st_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = 'CUI-less'
    note_file_name = read.textfile2list(file_dir_path)
    note_file_name = [item + ".txt" for item in note_file_name]

    normdir = notedir.replace('note', 'norm')

    input = []
    st = []
    st_all = []
    st_only_multiple = []
    st_only_single = []

    for note in note_file_name:
        norm = note.replace('txt', 'norm')
        conceptlist = process.load_concept(normdir + norm)
        for concepts in conceptlist:
            concept_cui = concepts[1]
            cui_st_list = semantic_type[concepts[1]]

            # elif "Pharmacologic Substance" in semantic_type[concepts[1]]:
            #     cui_st = ["Pharmacologic Substance"]
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

            st += cui_st_list
            st_all.append("_".join(cui_st_list))
            if len(cui_st_list) > 1:
                st_only_multiple.append("_".join(cui_st_list))
            else:
                st_only_single.append("_".join(cui_st_list))
    # print(Counter(st))
    # print(Counter(st_all))
    print(Counter(st_only_multiple))

    print(Counter(st_only_single))


analyze_st_freq("data/n2c2/train_dev/dev_file_list.txt",
                "data/n2c2/train_dev/train_note/")

analyze_st_freq("data/n2c2/test/test_file_list.txt",
                "data/n2c2/test/test_note/")
