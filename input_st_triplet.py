import os

import sklearn.metrics as metrics
from sklearn.metrics import classification_report

import process_funtions as process
import read_files as read


def update_pre_concept_info(pre_concept_info, note_id, concept_id, concept,
                            concept_text, token, pred_label):
    if note_id in pre_concept_info:
        if concept_id in pre_concept_info[note_id]:
            pre_concept_info[note_id][concept_id]["token"].append(token)
            pre_concept_info[note_id][concept_id]["gold_cui"].append(concept)
            pre_concept_info[note_id][concept_id]["pre_st"].append(pred_label)
            pre_concept_info[note_id][concept_id]["gold_text"].append(
                concept_text)

        else:
            pre_concept_info[note_id][concept_id] = {
                "token": [token],
                "gold_cui": [concept],
                "pre_st": [pred_label],
                "gold_text": [concept_text],
            }

    else:
        pre_concept_info[note_id] = {
            concept_id: {
                "token": [token],
                "gold_cui": [concept],
                "pre_st": [pred_label],
                "gold_text": [concept_text],
            }
        }

    return pre_concept_info


def input_st_raw(cui_file_path, gold_file_path, pre_file_path, output_path):
    cui_inputs = read.read_from_json(cui_file_path)
    inputs = read.read_from_tsv(gold_file_path)
    predictions = read.textfile2list(pre_file_path)
    pre_concept_info = {}
    for cui_input, input, prediction in zip(cui_inputs, inputs, predictions):
        cui_labels = cui_input[1]
        tokens = input[1].split()
        gold_labels = input[0].split()
        pred_labels = prediction.split()

        if len(tokens) != len(gold_labels) or len(gold_labels) != len(
                pred_labels):
            raise ValueError(
                "The length of predictions and the input is not equal!")

        for token, gold_label, pred_label, cui_label in zip(
                tokens, gold_labels, pred_labels, cui_labels):
            if gold_label[0] in ['B', 'I']:
                cui_label_content, note_id = cui_label.split("+++")

                _, concept_id, concept, text = cui_label_content.split("_")
                if pred_label[0] in ['B', 'I']:
                    pred_label = " ".join(pred_label.split("-")[1].split("_"))
                else:
                    pred_label = "None"

                pre_concept_info = update_pre_concept_info(
                    pre_concept_info, note_id, concept_id, concept, text,
                    token, pred_label)

    read.save_in_json(output_path, pre_concept_info)


# input_st_raw("data/n2c2/processed/raw/dev",
#              "data/n2c2/processed/input_tagger/sg/dev.tsv",
#              "data/n2c2/models/sg_0401_b32_e20/eval_predictions.txt",
#              "data/n2c2/processed/input_st/sg/raw/dev")


def process_input_query(file_dir_path, norm_path, raw_input_st_dir,
                        output_dir):

    file_names = read.textfile2list(file_dir_path)
    norm_file_names = [item + ".norm" for item in file_names]

    raw_input_st_info = read.read_from_json(raw_input_st_dir)

    semantic_type = read.read_from_json(
        "data/umls/cui_st_term_snomed_rxnorm_dict_all")
    semantic_type['CUI-less'] = ['CUI_less']

    semantic_group = read.read_from_json(
        "data/umls/cui_sgroup_term_snomed_rxnorm_dict_all")
    semantic_group['CUI-less'] = ['CUI_less']

    semantic_type_pre = []
    semantic_type_pre_list = []
    semantic_type_gold = []

    input_query_dev = []
    dev_norm = []

    for file_name in file_names:
        conceptlist = process.load_concept(
            os.path.join(norm_path, file_name + ".norm"))
        for concept_info in conceptlist:
            token = " ".join(
                raw_input_st_info[file_name][concept_info[0]]["token"])
            gold_cui = list(
                set(raw_input_st_info[file_name][concept_info[0]]["gold_cui"]))
            mention = list(
                set(raw_input_st_info[file_name][concept_info[0]]
                    ["gold_text"]))
            pre_st = raw_input_st_info[file_name][concept_info[0]]["pre_st"]
            # print(file_name, concept_info[0], token, mention, gold_cui,
            #       process.get_st_cui(gold_cui[0]), pre_st)
            semantic_type_pre_list.append(pre_st)
            semantic_type_pre.append(pre_st[-1])
            semantic_type_gold.append(
                process.get_sg_cui(semantic_group, gold_cui[0]))

            # print(file_name, concept_info[0], token, mention, gold_cui,
            #       process.get_st_cui(semantic_type,gold_cui[0]), pre_st)

            if len(mention) > 1 or mention[0] != token or len(gold_cui) > 1:
                raise ValueError("mention texts is more than one.")

            input_query_dev.append([token + " " + pre_st[-1]])
            dev_norm.append(gold_cui[0])

    labels = list(set(semantic_type_gold + semantic_type_pre))

    process.raw_accuracy_score(semantic_type_gold, semantic_type_pre_list)

    read.save_in_tsv(os.path.join(output_dir, "dev_query.tsv"),
                     input_query_dev)
    # read.save_in_json(os.path.join(output_dir, "dev_norm"), dev_norm)

    print(
        metrics.f1_score(semantic_type_gold,
                         semantic_type_pre,
                         average='micro'))

    print(
        classification_report(semantic_type_gold,
                              semantic_type_pre,
                              target_names=labels))


# process_input_query("data/n2c2/train_dev/dev_file_list.txt",
#                     "data/n2c2/train_dev/train_norm/",
#                     "data/n2c2/processed/input_st/raw/dev",
#                     "data/n2c2/processed/input_st/triplet/data/")

process_input_query("data/n2c2/train_dev/dev_file_list.txt",
                    "data/n2c2/train_dev/train_norm/",
                    "data/n2c2/processed/input_st/sg/raw/dev",
                    "data/n2c2/processed/input_st/sg/triplet/data/")
