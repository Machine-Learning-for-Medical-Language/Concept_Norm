import csv
import json
import os
import re

import process_funtions as process
import read_files as read


def build_dataset(input_path):

    train_input = read.textfile2list(input_path)

    train_input = [item.split("\t") for item in train_input]

    ontology = read.read_from_json("data/share/umls/cui_umls_for_share")

    print(len(ontology))

    # input = []
    processed = []
    unseen = []
    for mention, concept in train_input:
        concept = concept.replace(".", "")

        concept = concept.strip()
        if concept in ontology:
            processed.append([mention, concept])
        elif concept.lower() == "cui-less":
            processed.append([mention, "CUI-less"])
        else:
            processed.append([mention, ontology[0]])
            unseen.append(concept)
    print(unseen)
    print(len(unseen))
    print(len(list(set(unseen))))

    read.save_in_tsv("data/share/processed/data_raw/test.tsv", processed)


# input_path = "data/share/raw/train.txt"
# input_path = "data/share/raw/dev.txt"
# input_path = "data/share/raw/test.txt"
# build_dataset(input_path)


def data_st():
    input = read.read_from_tsv("data/share/processed/data_raw/test.tsv")

    semantic_type = read.read_from_json("data/share/umls/cui_share_st")
    semantic_type['CUI-less'] = ['CUI_less']

    input_new = []

    for [mention, concept] in input:
        input_st = '_'.join(
            process.get_st_cui(semantic_type, concept).split(" "))
        input_synonym_new = "<e> " + mention + " </e>"

        input_new.append([input_st, concept, input_synonym_new])
    read.save_in_tsv("data/share/processed/data/test.tsv", input_new)


# data_st()


def combine_train_dev():
    umls = read.read_from_tsv("data/share/umls/ontology.tsv")

    train = read.read_from_tsv("data/share/processed/data/train.tsv")

    train_new = umls + train * 20

    read.save_in_tsv("data/share/processed/umls+data/train.tsv", train_new)


# combine_train_dev()


def share_input():
    ontology = read.read_from_tsv("data/share/umls/ontology.tsv")
    train = read.read_from_tsv("data/share/processed/data/train.tsv")
    dev = read.read_from_tsv("data/share/processed/data/dev.tsv")

    ontology = ontology + train + dev

    cuis = read.read_from_json("data/share/umls/cui_umls_for_share")
    norm_mention = {}
    for idx, [_, norm, mention] in enumerate(ontology):
        read.add_dict(norm_mention, norm, mention)

    mentions = []
    idx = 0
    cui_mention_idx = {}

    for cui in cuis:
        cui_mentions = list(set(norm_mention[cui]))
        mentions += cui_mentions
        end = idx + len(cui_mentions)
        cui_mention_idx[cui] = (idx, end)
        idx = end
    mentions = [[syn] for syn in mentions]

    print(len(cui_mention_idx))

    read.save_in_tsv("data/share/umls_concept/ontology_synonyms.tsv", mentions)
    read.save_in_json("data/share/umls_concept/ontology_concept_synonyms_idx",
                      cui_mention_idx)

    return mentions, cui_mention_idx


# share_input()
