import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import process_funtions as process
import read_files as read


def analyze_cn_top1(dev, cui_file_path, input_file_path):

    semantic_type = read.read_from_json(
        "data/umls/cui_st_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    cui_synonyms = read.read_from_json(
        "data/n2c2/triplet_network/con_norm/ontology_concept_synonyms_dict")
    cui_synonyms['CUI-less'] = ['CUI_less']

    if dev == True:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention/train.tsv")
    else:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention/train.tsv"
        ) + read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention/dev.tsv")

    train_cui = {}
    for item in train_input:
        train_cui = read.add_dict(train_cui, item[1], item[2])
    # train_cui = [item[1] for item in train_input]

    dev_pre = read.textfile2list(cui_file_path)

    dev_input = read.read_from_tsv(input_file_path)

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
        st, cui, mention = input

        st_pre = '_'.join(
            process.get_st_cui(semantic_type, pre_cui).split(' '))

        if st_pre == st:
            count_st += 1

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
            print(cui, st, mention)
            print()
            print(cui_synonyms[cui])
            print()
            print(pre_cui, st_pre, cui_synonyms[pre_cui])
            print()
            print()
            print()
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


# cui_file_path = "data/n2c2/models/umls+data_c4255/st_joint_eval_predictions.txt"
# input_file_path = "data/n2c2/processed/input_joint/umls+data/dev.tsv"
# analyze_cn_top1(True, cui_file_path, input_file_path)

# cui_file_path = "data/n2c2/models/e20_b16_s128_5e5/"
# input_file_path = "data/n2c2/processed/input_joint/st/test.tsv"
# analyze_cn_top1(False, cui_file_path, input_file_path)


def analyze_cn_topk(dev, st_file_path, cui_file_path, input_file_path):

    semantic_type = read.read_from_json(
        "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    cui_synonyms = read.read_from_json(
        "data/n2c2/triplet_network/con_norm/ontology_concept_synonyms")
    cui_synonyms['CUI-less'] = ['CUI_less']

    if dev == True:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention_sg/train.tsv")
    else:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention_sg/train.tsv"
        ) + read.read_from_tsv(
            "data/n2c2/processed/input_joint/mention_sg/dev.tsv")

    train_cui = {}
    for item in train_input:
        train_cui = read.add_dict(train_cui, item[1], item[2])
    # train_cui = [item[1] for item in train_input]

    dev_pre_cui = read.textfile2list(cui_file_path + ".txt")
    dev_pre_cui = [item.split(" ") for item in dev_pre_cui]
    dev_pre_score_cui = np.load(cui_file_path + ".npy")

    dev_pre_st = read.textfile2list(st_file_path + ".txt")
    dev_pre_st = [item.split(" ") for item in dev_pre_st]

    dev_pre_score_st = np.load(st_file_path + ".npy")

    dev_input = read.read_from_tsv(input_file_path)

    count_st = 0

    count_all = len(dev_input)
    count = 0
    countnot = 0
    countst = 0

    count_see = 0
    count_see_all = 0

    count_unsee = 0
    count_unsee_pre_seen = 0
    count_unsee_all = 0

    count_cuiless = 0
    count_cuiless_all = 0

    count_st = 0

    output = []

    for index, [pre_cuis, input] in enumerate(zip(dev_pre_cui, dev_input)):
        st, cui, mention = input

        st_pre = [
            '_'.join(process.get_sg_cui(semantic_type, item).split(' '))
            for item in pre_cuis
        ]

        if st_pre == st:
            count_st += 1

        if cui in pre_cuis[:1]:
            count += 1
            # print(cui, st, mention)
            # print()
            # print(cui_synonyms[cui])
            # print()
            # # print(pre_cui, st_pre, cui_synonyms[pre_cui])
            # print()
            # print()
            # print()
        if cui in pre_cuis[:3] and cui not in pre_cuis[:
                                                       1] and cui != "CUI-less":
            print("Real data:", mention, cui, st)
            print(cui_synonyms[cui])
            print()

            countnot += 1
            st = [
                process.get_sg_cui(semantic_type, item)
                for item in pre_cuis[:2]
            ]
            st_0 = process.get_sg_cui(semantic_type, pre_cuis[0])
            st_0_pre = dev_pre_st[index][0]
            st_0_pre_score = dev_pre_score_st[index][0]

            st_1 = process.get_sg_cui(semantic_type, pre_cuis[1])
            st_1_pre = dev_pre_st[index][1]
            st_1_pre_score = dev_pre_score_st[index][1]

            if len(list(set(st))) == 2:
                countst += 1

            sts = [st_0, st_1]
            sts_pre = [st_0_pre, st_1_pre]
            sts_pre_score = [st_0_pre_score, st_1_pre_score]

            print("***prediction***")
            for cui_idx, cui_pre in enumerate(pre_cuis[:2]):
                score = dev_pre_score_cui[index][cui_idx]
                print(cui_pre, score, st[cui_idx], cui_synonyms[cui_pre],
                      sts_pre[cui_idx], sts_pre_score[cui_idx])

                print()
            print("***Done***")
            print()
            print()
    #     pre_cui = pre_cuis[0]
    #     if cui == 'CUI-less':
    #         count_cuiless_all += 1
    #         if cui == pre_cui:
    #             count_cuiless += 1

    #     else:

    #         if cui in train_cui:
    #             count_see_all += 1
    #             if cui == pre_cui:
    #                 count_see += 1

    #         else:
    #             count_unsee_all += 1
    #             if cui == pre_cui:
    #                 count_unsee += 1

    #                 # print(cui, st, mention)
    #                 # print()
    #                 # print(cui_synonyms[cui])
    #                 # print()
    #                 # print(pre_cui, st_pre, cui_synonyms[pre_cui])
    #                 # print()
    #                 # print()
    #                 # print()
    #             else:
    #                 if pre_cui in train_cui:
    #                     count_unsee_pre_seen += 1
    #                 #     print("special notification......")
    #                 #     print(train_cui[pre_cui])

    #     #             # print(cui, st, mention)
    #     #             # print()
    #     #             # print(list(set(cui_synonyms[cui])))
    #     #             # print()
    #     #             # print(pre_cui, st_pre, list(set(cui_synonyms[pre_cui])))
    #     #             # print()
    #     #             # print()
    #     #             # print()

    #     #             #     print(cui, st, mention)
    #     #             #     print()
    #     #             #     print(cui_synonyms[cui])
    #     #             #     print()
    #     #             #     print(pre_cui, st_pre, cui_synonyms[pre_cui])
    #     #             #     print()
    #     #             #     print()
    #     #             #     print()
    print("acc", count / count_all, "ambigupus", countst / countnot,
          countst / count_all)
    # print("cuiless", count_cuiless / count_cuiless_all)

    # print("seen", count_see / count_see_all, "unseen",
    #       count_unsee / count_unsee_all, "unseen gold truth but seen pred",
    #       count_unsee_pre_seen / count_unsee_all)
    # print("st", count_st / (count_all - count_cuiless))


cui_folder_path = "data/n2c2/models/best_0_checkpoint-1477//"
input_folder_path = "data/n2c2/processed/input_joint/umls+data_sg/"
dev = True

if dev == True:
    cui_file_path = cui_folder_path + "cn_joint_eval_predictions"
    st_file_path = cui_folder_path + "st_joint_eval_predictions"
    input_file_path = input_folder_path + "dev.tsv"
else:
    cui_file_path = cui_folder_path + "cn_joint_test_predictions"
    st_file_path = cui_folder_path + "st_joint_test_predictions"
    input_file_path = input_folder_path + "test.tsv"
analyze_cn_topk(dev, st_file_path, cui_file_path, input_file_path)
