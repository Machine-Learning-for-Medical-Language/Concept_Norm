import csv
import json
import os
import re

import process_funtions as process
import read_files as read


def mapping_concept_context(concept_file, document_file):
    document_info = {}
    with open(document_file, 'r') as f:
        offset = 0

        conceptlist = process.load_concept(concept_file)
        for index, line in enumerate(f):
            sent_info = {}

            sent_info["text"] = line
            sent_info["offset"] = [offset, offset + len(line)]
            token_spans = [(m.start(), m.end())
                           for m in re.finditer(r'\S+', line.strip())]
            sent_info["token_spans"] = token_spans
            sent_info["tokens"] = [
                line.strip()[token_span[0]:token_span[1]]
                for token_span in token_spans
            ]
            concepts = []

            for concept_info in conceptlist:
                concept_id = concept_info[0]
                concept = concept_info[1]
                concept_span = concept_info[2]
                if concept_span[0][0] >= offset and concept_span[-1][
                        1] <= offset + len(line):
                    concept_offsets = [(item[0] - offset, item[1] - offset)
                                       for item in concept_span]
                    concept_tokens = [
                        line[concept_offset[0]:concept_offset[1]]
                        for concept_offset in concept_offsets
                    ]
                    concepts.append(
                        [concept_id, concept, concept_offsets, concept_tokens])

            sent_info["concepts"] = concepts

            document_info[str(index)] = sent_info
            offset += len(line)

    return document_info


def build_dataset(file_dir_path, notedir, out_file_name):

    note_file_name = read.textfile2list(file_dir_path)
    note_file_name = [item + ".txt" for item in note_file_name]

    normdir = notedir.replace('note', 'norm')

    input = []

    for note in note_file_name:
        norm = note.replace('txt', 'norm')
        sentence_infos = mapping_concept_context(normdir + norm,
                                                 notedir + note)
        for index in range(len(sentence_infos)):
            sentence_info = sentence_infos[str(index)]
            text = sentence_info["text"]
            tokens = sentence_info["tokens"]
            token_spans = sentence_info["token_spans"]
            concepts = sentence_info["concepts"]

            tokens, token_spans = process.match_token_span(
                text, tokens, token_spans, concepts)

            contain_discontinous = False
            for concept in concepts:
                if concept[0] == "N055" and "0358" in note:
                    print(1)

                concept_span = concept[2]
                if len(concept_span) >= 2:
                    contain_discontinous = True
                    break

            # if contain_discontinous is False:
            #     tokens, tags = process.create_tagging(text, tokens,
            #                                           token_spans, concepts,
            #                                           note)

            # else:
            tokens, tags = process.create_tagging_discontinuous(
                text, tokens, token_spans, concepts, note)

            # print(text, tokens, tags)
            # mention = concept[0]
            # cui = concept[1]
            # spans = concept[2]

            # cui_st = semantic_type[cui]
            # if len(cui_st) > 1:
            #     print(text, cui, mention, cnormui_st)

            ### todo: get the index for the concepts
            input.append([tokens, tags])

            print(tokens, tags)

    read.save_in_json(out_file_name, input)


build_dataset("data/n2c2/train_dev/dev_file_list.txt",
              "data/n2c2/train_dev/train_note/",
              out_file_name='data/n2c2/processed/raw/dev_all')

build_dataset("data/n2c2/train_dev/train_file_list.txt",
              "data/n2c2/train_dev/train_note/",
              out_file_name='data/n2c2/processed/raw/train_all')

### Fixed to do for discontinuous mentions in test set
build_dataset("data/n2c2/test/test_file_list.txt",
              "data/n2c2/test/test_note/",
              out_file_name='data/n2c2/processed/raw/test_all')
