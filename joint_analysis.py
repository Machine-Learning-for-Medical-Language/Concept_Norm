import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import process_funtions as process
import read_files as read


def analyze_cn_topk(dev, cui_file_path, cui_sg_file_path,
                    cui_per_candidate_file_path, context_sg_file_path,
                    input_file_path):

    semantic_type = read.read_from_json(
        "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    cui_synonyms = read.read_from_json(
        "data/n2c2/triplet_network/con_norm/ontology_concept_synonyms")
    cui_synonyms['CUI-less'] = ['CUI_less']

    semantic_type_label = read.read_from_json("data/umls/umls_sg")

    st_labels = []
    for label in semantic_type_label:
        label_new = '_'.join(label.split(' '))
        st_labels.append(label_new)

    st_labels.append('CUI_less')
    st_idx = {item: idx for idx, item in enumerate(st_labels)}

    if dev == True:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/sentence_mention_st/train.tsv")
    else:
        train_input = read.read_from_tsv(
            "data/n2c2/processed/input_joint/sentence_mention_st/train.tsv"
        ) + read.read_from_tsv(
            "data/n2c2/processed/input_joint/sentence_mention_st/dev.tsv")

    train_cui = {}
    for item in train_input:
        train_cui = read.add_dict(train_cui, item[1], item[2])
    # train_cui = [item[1] for item in train_input]

    dev_pre_cui = read.textfile2list(cui_file_path + ".txt")
    dev_pre_cui = [item.split(" ") for item in dev_pre_cui]
    dev_pre_score_cui = np.load(cui_file_path + ".npy")

    dev_pre_cui_sg = read.textfile2list(cui_sg_file_path + ".txt")
    dev_pre_cui_sg = [item.split(" ") for item in dev_pre_cui_sg]
    dev_pre_score_sg = np.load(cui_sg_file_path + ".npy")

    # dev_pre_score_sg_idx = np.argsort(dev_pre_score_sg, axis=-1)
    # dev_pre_score_sg_idx = dev_pre_score_sg_idx[:, ::-1]

    dev_pre_cui_per_sg_candidate = read.textfile2list(
        cui_per_candidate_file_path + ".txt")
    dev_pre_cui_per_sg_candidate = [
        item.split(" ") for item in dev_pre_cui_per_sg_candidate
    ]

    dev_pre_context_sg = read.textfile2list(context_sg_file_path + ".txt")
    dev_pre_context_sg = [item.split(" ") for item in dev_pre_context_sg]
    dev_pre_context_sg_idx = [[st_idx[sg] for sg in item]
                              for item in dev_pre_context_sg]
    dev_pre_context_sg_score = np.load(context_sg_file_path + ".npy")

    dev_input = read.read_from_tsv(input_file_path)

    count_all = len(dev_input)
    count_both = 0
    count_mention_no_context = 0
    count_context_no_mention = 0
    count_neither = 0

    count_mention_sg_recall = 0
    count_mention_recall = 0

    count_rules = 0
    count_rules_sg = 0

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

    st_gold = []
    st_mention_pre = []
    st_context_pre = []

    for index, [cuis_pre_mention,
                input] in enumerate(zip(dev_pre_cui, dev_input)):
        st, cui, mention, context = input
        st_gold.append(st)

        cui_sg_pre_noclassifier = [
            '_'.join(process.get_sg_cui(semantic_type, item).split(' '))
            for item in cuis_pre_mention
        ]
        st_mention_pre.append(cui_sg_pre_noclassifier[0])

        cui_sg_pre_noclassifier_score = dev_pre_score_cui[index]

        cui_s_sg_pre = dev_pre_cui_sg[index]
        cui_s_sg_pre_score = dev_pre_score_sg[index]

        context_s_sg_pre = dev_pre_context_sg[index]
        st_context_pre.append(context_s_sg_pre[0])

        context_s_sg_pre_score = dev_pre_context_sg_score[index]

        cuis_pre_sg_candidate = dev_pre_cui_per_sg_candidate[index]

        cuis_pre_context = [
            cuis_pre_sg_candidate[item]
            for item in dev_pre_context_sg_idx[index]
        ]

        if cui in cuis_pre_context[:17]:
            count_mention_recall += 1

        if st in cui_s_sg_pre[:3]:
            count_mention_sg_recall += 1

        # print(cuis_pre_mention, cuis_pre_context)

        if cui == cuis_pre_mention[0] and cui == cuis_pre_context[0]:
            count_both += 1

        elif cui == cuis_pre_mention[0] and cui != cuis_pre_context[0]:
            count_mention_no_context += 1

        elif cui != cuis_pre_mention[0] and cui == cuis_pre_context[0]:
            count_context_no_mention += 1

        else:
            count_neither += 1
            # print(
            #     st,
            #     mention,
            #     context,
            # )
            # print(cui_sg_pre_noclassifier[:5],
            #       cui_sg_pre_noclassifier_score[:5])
            # print(cui_s_sg_pre[:5], cui_s_sg_pre_score[:5])
            # print(
            #     context_s_sg_pre[:5],
            #     context_s_sg_pre_score[:5],
            # )
            # print(0)

        if cui_sg_pre_noclassifier[0] == "CUI-less":
            cui_pre = cuis_pre_mention[0]

        elif cui_sg_pre_noclassifier[0] == cui_sg_pre_noclassifier[1]:
            cui_pre = cuis_pre_mention[0]

        elif cui_sg_pre_noclassifier[0] in [
                "Chemicals_&_Drugs", "Concepts_&_Ideas", "Devices",
                "Phenomena", "Physiology"
        ] and context_s_sg_pre[0] == "Procedures":
            cui_pre = cuis_pre_context[0]

        else:
            cui_pre = cuis_pre_mention[0]

        if cui_pre == cui:
            count_rules += 1

        st_pre = '_'.join(
            process.get_sg_cui(semantic_type, cui_pre).split(' '))
        if st_pre == st:
            count_rules_sg += 1

        #     if st_pre == st:
        #         count_st += 1

        # if cui in pre_cuis_new[:1]:
        #     count += 1
    # print(cui, st, mention)
    # print()
    # print(cui_synonyms[cui])
    # print()
    # # print(pre_cui, st_pre, cui_synonyms[pre_cui])
    # print()
    # print()
    # print()
    # if cui in pre_cuis[:
    #                        3] and cui not in pre_cuis[:
    #                                                       1] and cui != "CUI-less":
    #     print("Real data:", mention, cui, st)
    #     print(cui_synonyms[cui])
    #     print()

    #     countnot += 1
    #     st = [
    #         process.get_sg_cui(semantic_type, item)
    #         for item in pre_cuis[:2]
    #     ]
    #     st_0 = process.get_sg_cui(semantic_type, pre_cuis[0])
    #     st_0_pre = dev_pre_st[index][0]
    #     st_0_pre_score = dev_pre_score_st[index][0]

    #     st_1 = process.get_sg_cui(semantic_type, pre_cuis[1])
    #     st_1_pre = dev_pre_st[index][1]
    #     st_1_pre_score = dev_pre_score_st[index][1]

    #     if len(list(set(st))) == 2:
    #         countst += 1

    #     sts = [st_0, st_1]
    #     sts_pre = [st_0_pre, st_1_pre]
    #     sts_pre_score = [st_0_pre_score, st_1_pre_score]

    #     print("***prediction***")
    #     for cui_idx, cui_pre in enumerate(pre_cuis[:2]):
    #         # score = dev_pre_score_cui[index][cui_idx]
    #         # print(cui_pre, score, st[cui_idx], cui_synonyms[cui_pre],
    #         #       sts_pre[cui_idx], sts_pre_score[cui_idx])
    #         print(cui_pre, st[cui_idx], cui_synonyms[cui_pre],
    #               sts_pre[cui_idx], sts_pre_score[cui_idx])

    #         print()
    # print("***Done***")
    # print()
    # print()
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
    # print("acc", count / count_all)  #, "ambigupus", countst / countnot,
    #   countst / count_all)
    # print("cuiless", count_cuiless / count_cuiless_all)

    # print("seen", count_see / count_see_all, "unseen",
    #       count_unsee / count_unsee_all, "unseen gold truth but seen pred",
    #       count_unsee_pre_seen / count_unsee_all)
    # print("st", count_st / (count_all - count_cuiless))

    conf_matrix_all = confusion_matrix(st_gold,
                                       st_mention_pre,
                                       labels=st_labels)



    conf_matrix_all_new = []
    conf_matrix_all_new.append([''] + st_labels)
    for idx, [score, label] in enumerate(zip(conf_matrix_all, st_labels)):
        conf_matrix_all_new.append([label] + list(score))
    print(conf_matrix_all_new)

    read.save_in_tsv('./confusion_matrix_mention_dev.tsv', conf_matrix_all_new)

    print(count_all, count_rules_sg, "top k cuis after rules", count_rules,
          "top k cuis using context to rank", count_mention_recall,
          "top k sgs using context to rank", count_mention_sg_recall,
          count_both, count_mention_no_context, count_context_no_mention,
          count_neither)


cui_folder_path = "data/n2c2/models/best_0_checkpoint-1477/"
context_folder_path = "data/n2c2/models/mention_only/"
input_folder_path = "data/n2c2/processed/input_joint/sentence_mention_st/"
dev = False

if dev == True:
    cui_file_path = cui_folder_path + "cn_joint_eval_predictions"
    cui_sg_file_path = cui_folder_path + "st_joint_eval_predictions"
    cui_per_candidate_file_path = context_folder_path + "cn_joint_eval_predictions"
    context_sg_file_path = context_folder_path + "st_joint_eval_predictions"

    input_file_path = input_folder_path + "dev.tsv"
else:
    cui_file_path = cui_folder_path + "cn_joint_test_predictions"
    cui_sg_file_path = cui_folder_path + "st_joint_test_predictions"
    cui_per_candidate_file_path = context_folder_path + "cn_joint_test_predictions"
    context_sg_file_path = context_folder_path + "st_joint_test_predictions"

    input_file_path = input_folder_path + "test.tsv"

analyze_cn_topk(dev, cui_file_path, cui_sg_file_path,
                cui_per_candidate_file_path, context_sg_file_path,
                input_file_path)

# analyze_cn_topk(dev, st_file_path, cui_file_path, input_file_path)
