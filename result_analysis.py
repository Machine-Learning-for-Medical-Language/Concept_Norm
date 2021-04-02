import os
from collections import Counter

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import read_files as read


def analyze_token_level_confusion(cui_file_path, gold_file_path, pre_file_path,
                                  output_path):
    inputs = read.read_from_tsv(gold_file_path)
    predictions = read.textfile2list(pre_file_path)
    cui_inputs = read.read_from_json(cui_file_path)

    analysis_token = []
    analysis_all = []

    analysis_token.append(['input', 'gold', 'prediction', ' indicator', 'cui'])
    analysis_all.append(['input', 'gold', 'prediction', ' indicator', 'cui'])

    gold_o_e = []
    pre_o_e = []

    gold_all = []
    pre_all = []

    for cui_input, input, prediction in zip(cui_inputs, inputs, predictions):
        cui_labels = cui_input[1]
        tokens = input[1].split()
        gold_labels = input[0].split()
        pred_labels = prediction.split()

        if len(tokens) != len(gold_labels) or len(gold_labels) != len(
                pred_labels):
            raise ValueError(
                "The length of predictions and the input is not equal!")
        # not_equal = False

        for token, gold_label, pred_label, cui_label in zip(
                tokens, gold_labels, pred_labels, cui_labels):
            if gold_label[0] in ['B', 'I']:
                gold_label_token = 'E'
                gold_all.append(gold_label[2:])
            else:
                gold_label_token = 'O'
                gold_all.append('O')

            if pred_label[0] in ['B', 'I']:
                pred_label_token = 'E'
                pre_all.append(pred_label[2:])
            else:
                pred_label_token = 'O'
                pre_all.append('O')

            gold_o_e.append(gold_label_token)
            pre_o_e.append(pred_label_token)

            if gold_label_token != pred_label_token:
                analysis_token.append([
                    token, gold_label_token, pred_label_token, "wrong",
                    cui_label
                ])
            else:
                analysis_token.append(
                    [token, gold_label_token, pred_label_token, "", cui_label])

            if gold_label != pred_label:
                analysis_all.append(
                    [token, gold_label, pred_label, "wrong", cui_label])
            else:
                analysis_all.append(
                    [token, gold_label, pred_label, "", cui_label])

        analysis_token.append(["---"] * 4)
        analysis_all.append(["---"] * 4)

    read.save_in_tsv(os.path.join(output_path, 'tagging_o_e_analysis.tsv'),
                     analysis_token)
    read.save_in_tsv(os.path.join(output_path, 'tagging_all_analysis.tsv'),
                     analysis_all)

    # print(set(gold_o_e), set(pre_o_e))
    print(Counter(gold_o_e))
    print(metrics.f1_score(gold_o_e, pre_o_e, pos_label="E"))
    conf_matrix = confusion_matrix(gold_o_e, pre_o_e, labels=['E', 'O'])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=['E', 'O'])
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(output_path, 'e_o_confusion_matrix.png'))

    label_all = list(set(gold_all + pre_all))
    conf_matrix_all = confusion_matrix(gold_all, pre_all, labels=label_all)
    conf_matrix_all_new = []
    conf_matrix_all_new.append([''] + label_all)
    for idx, [score, label] in enumerate(zip(conf_matrix_all, label_all)):
        conf_matrix_all_new.append([label] + list(score))

    read.save_in_tsv(os.path.join(output_path, 'all_confusion_matrix.tsv'),
                     conf_matrix_all_new)


analyze_token_level_confusion(
    "data/n2c2/processed/raw/dev",
    "data/n2c2/processed/input_tagger/sg/dev.tsv",
    "data/n2c2/models/sg_0401_b32_e20/eval_predictions.txt",
    "data/n2c2/results/sg/")


def generate_output(gold_file_path, pre_file_path, output_path):
    inputs = read.read_from_tsv(gold_file_path)
    predictions = read.textfile2list(pre_file_path)

    analysis_all = []

    analysis_all.append(['input', 'prediction'])
    notes_dir = ['note1.txt', 'note2.txt', 'note3.txt']
    note_idx = 0

    for input, prediction in zip(inputs, predictions):
        tokens = input[1].split()

        pred_labels = prediction.split()

        if len(tokens) != len(pred_labels):
            raise ValueError(
                "The length of predictions and the input is not equal!")
        # not_equal = False

        for token, pred_label in zip(tokens, pred_labels):
            if token != "EOS":
                analysis_all.append([token, pred_label])

        if token == "EOS":

            analysis_all.append(
                ["End of " + notes_dir[note_idx], "----------"])
            analysis_all.append(["", ""])
            analysis_all.append(["", ""])
            note_idx += 1
        else:
            analysis_all.append(["<EOS>", "----"])

    read.save_in_tsv(os.path.join(output_path, 'tagging_analysis.tsv'),
                     analysis_all)


# generate_output(
#     "/home/dongfangxu/Projects/Concept_Norm/data/covid_data/input/test.tsv",
#     "/home/dongfangxu/Projects/Concept_Norm/data/covid_data/output/test_predictions.txt",
#     "/home/dongfangxu/Projects/Concept_Norm/data/covid_data/results/")
