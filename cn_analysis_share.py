import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import process_funtions as process
import read_files as read


def analyze_cn_top1(dev, cui_file_path, input_file_path):

    # semantic_type = read.read_from_json(
    #     "data/umls/cui_st_term_snomed_rxnorm_dict_all")
    # semantic_type['CUI-less'] = ['CUI_less']

    cui_synonyms = read.read_from_json(
        "data/share/umls/ontology_concept_synonyms")
    cui_synonyms['CUI-less'] = ['CUI_less']

    if dev == True:
        train_input = read.read_from_tsv("data/share/processed/data/train.tsv")
    else:
        train_input = read.read_from_tsv(
            "data/share/processed/data/train.tsv") + read.read_from_tsv(
                "data/share/processed/data/dev.tsv")

    train_cui = {}
    for item in train_input:
        train_cui = read.add_dict(train_cui, item[1], item[2])
    # train_cui = [item[1] for item in train_input]

    dev_pre = read.textfile2list(cui_file_path)
    dev_pre = [item.split(" ") for item in dev_pre]

    # dev_input = read.read_from_tsv(input_file_path)

    dev_input = read.textfile2list(input_file_path)

    dev_input = [item.split("\t") for item in dev_input]

    count_st = 0

    count_all = len(dev_input)
    count = 0

    count_see = 0
    count_see_all = 0

    count_unsee = 0
    count_unsee_pre_seen = 0
    count_unsee_all = 0

    count_cuiless = 0
    count_cuiless_all = 0

    count_st = 0

    output = []

    for pre_cui, input in zip(dev_pre, dev_input):
        # st, cui, mention = input
        mention, cui = input
        pre_cui = pre_cui[0]
        # st_pre = '_'.join(
        #     process.get_st_cui(semantic_type, pre_cui).split(' '))

        # if st_pre == st:
        #     count_st += 1

        if cui == pre_cui:
            count += 1
            # print(cui, st, mention)
            # print()
            # print(cui_synonyms[cui])
            # print()
            # print(pre_cui, st_pre, cui_synonyms[pre_cui])
            # print()
            # print()
            # print()
        else:
            print(cui, mention)
            print()
            if cui in cui_synonyms:
                print(cui_synonyms[cui])
            print()
            print(pre_cui, cui_synonyms[pre_cui])
            print()
            print()
            print("________________")
        if cui == 'CUI-less':
            count_cuiless_all += 1
            if cui == pre_cui:
                count_cuiless += 1

        else:

            if cui in train_cui:
                count_see_all += 1
                if cui == pre_cui:
                    count_see += 1

            else:
                count_unsee_all += 1
                if cui == pre_cui:
                    count_unsee += 1

                    # print(cui, st, mention)
                    # print()
                    # print(cui_synonyms[cui])
                    # print()
                    # print(pre_cui, st_pre, cui_synonyms[pre_cui])
                    # print()
                    # print()
                    # print()
                else:
                    if pre_cui in train_cui:
                        count_unsee_pre_seen += 1
                    #     print("special notification......")
                    #     print(train_cui[pre_cui])

                    # print(cui, st, mention)
                    # print()
                    # print(list(set(cui_synonyms[cui])))
                    # print()
                    # print(pre_cui, st_pre, list(set(cui_synonyms[pre_cui])))
                    # print()
                    # print()
                    # print()

                    #     print(cui, st, mention)
                    #     print()
                    #     print(cui_synonyms[cui])
                    #     print()
                    #     print(pre_cui, st_pre, cui_synonyms[pre_cui])
                    #     print()
                    #     print()
                    #     print()
    print(
        "acc",
        count / count_all,
        "cuiless",
        # )
        count_cuiless / count_cuiless_all)
    print("seen", count_see / count_see_all, "unseen",
          count_unsee / count_unsee_all, "unseen gold truth but seen pred",
          count_unsee_pre_seen / count_unsee_all)
    print("st", count_st / (count_all - count_cuiless))


cui_file_path = "data/share/outputs/1_triplet_ontology+train_all_e20_b400_seq16_5e5_sc45_m0.35/st_joint_test_predictions.txt"
input_file_path = "data/share/raw/test.txt"
analyze_cn_top1(True, cui_file_path, input_file_path)

# cui_file_path = "data/n2c2/models/e20_b16_s128_5e5/"
# input_file_path = "data/n2c2/processed/input_joint/st/test.tsv"
# analyze_cn_top1(False, cui_file_path, input_file_path)



