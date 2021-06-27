import os
import re
import shutil
from pathlib import Path

import bioc

import read_files as read


def get_st_train_files_from_i2b2():
    train_file_list = read.textfile2list(
        "data/n2c2/train_dev/train_file_list.txt")
    dev_file_list = read.textfile2list("data/n2c2/train_dev/dev_file_list.txt")
    test_file_list = read.textfile2list("data/n2c2/test/test_file_list.txt")
    # print("train: ", train_file_list)
    # print("dev: ", dev_file_list)
    # print("test: ", test_file_list)

    n2c2 = train_file_list + dev_file_list + test_file_list

    train_partners = [
        item.replace(".txt", "")
        for item in os.listdir("data/i2b2_2010/train/partners/txt/")
    ]

    train_beth = [
        item.replace(".txt", "")
        for item in os.listdir("data/i2b2_2010/train/beth/txt/")
    ]
    test = [
        item.replace(".txt", "")
        for item in os.listdir("data/i2b2_2010/test/txt/")
    ]

    n2c2 = n2c2 + [".DS_Store"]

    i2b2_train_beth = [item for item in train_beth if item not in n2c2]

    i2b2_train_partner = [item for item in train_partners if item not in n2c2]

    i2b2_test = [item for item in test if item not in n2c2]

    print(len(i2b2_train_beth + i2b2_train_partner + i2b2_test))

    file_names = [i2b2_train_beth, i2b2_train_partner, i2b2_test]
    source_folders = ["train/beth", "train/partners", "test"]
    for idx, files in enumerate(file_names):
        source_folder = source_folders[idx]
        read.create_folder("data/i2b2_2010/semantic_group/train/concept/" +
                           files[0] + ".con")
        read.create_folder("data/i2b2_2010/semantic_group/train/txt/" +
                           files[0] + ".con")
        for file in files:
            # print(file)
            shutil.copy(
                "data/i2b2_2010/" + source_folder + "/concept/" + file +
                ".con",
                "data/i2b2_2010/semantic_group/train/concept/" + file + ".con")

            shutil.copy(
                "data/i2b2_2010/" + source_folder + "/txt/" + file + ".txt",
                "data/i2b2_2010/semantic_group/train/txt/" + file + ".txt")


# get_st_train_files_from_i2b2()


def read_text(pathname):
    with open(pathname) as fp:
        text = fp.read()
    sentences = []
    offset = 0
    for sent in text.split('\n'):
        sentence = bioc.BioCSentence()
        sentence.infons['filename'] = pathname.stem
        sentence.offset = offset
        sentence.text = sent
        sentences.append(sentence)
        i = 0
        for m in re.finditer('\S+', sent):
            if i == 0 and m.start() != 0:
                # add fake
                ann = bioc.BioCAnnotation()
                ann.id = f'a{i}'
                ann.text = ''
                ann.add_location(bioc.BioCLocation(offset, 0))
                sentence.add_annotation(ann)
                i += 1
            ann = bioc.BioCAnnotation()
            ann.id = f'a{i}'
            ann.text = m.group()
            ann.add_location(
                bioc.BioCLocation(m.start() + offset, len(m.group())))
            sentence.add_annotation(ann)
            i += 1
        offset += len(sent) + 1
    return sentences


def _get_ann_offset(sentences, match_obj, start_line_group, start_token_group,
                    end_line_group, end_token_group, text_group):
    assert match_obj.group(start_line_group) == match_obj.group(end_line_group)
    sentence = sentences[int(match_obj.group(start_line_group)) - 1]

    start_token_idx = int(match_obj.group(start_token_group))
    end_token_idx = int(match_obj.group(end_token_group))
    start = sentence.annotations[start_token_idx].total_span.offset
    end = sentence.annotations[end_token_idx].total_span.end
    text = match_obj.group(text_group)

    actual = sentence.text[start - sentence.offset:end -
                           sentence.offset].lower()
    expected = text.lower()
    assert actual == expected, 'Cannot match at %s:\n%s\n%s\nFind: %r, Matched: %r' \
                               % (
                               sentence.infons['filename'], sentence.text, match_obj.string, actual,
                               expected)
    return start, end, text


def generate_text_concept_input():
    file_name = [
        item.replace(".txt", "")
        for item in os.listdir("data/i2b2_2010/semantic_group/train/txt/")
    ]

    pattern = re.compile(
        r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"(\|\|a="(.*?)")?')

    input_partial = []
    input_full = []
    for file in file_name:
        sentences = read_text(
            Path("data/i2b2_2010/semantic_group/train/txt/" + file + ".txt"))
        annotations = read.textfile2list(
            "data/i2b2_2010/semantic_group/train/concept/" + file + ".con")
        pathname = Path("data/i2b2_2010/semantic_group/train/concept/" + file +
                        ".con")
        anns = []
        for i, annotation in enumerate(annotations):
            annotation = annotation.strip()
            m = pattern.match(annotation)
            assert m is not None

            start, end, text = _get_ann_offset(sentences, m, 2, 3, 4, 5, 1)

            entity = ["<e>", text, "</e>"]

            sentence = sentences[int(m.group(2)) - 1]
            if int(m.group(2)) - 1 <= 0:
                sentence_before = ""
            else:
                sentence_before = sentences[int(m.group(2)) - 2].text

            if int(m.group(2)) >= len(sentences):
                sentence_after = ""
            else:
                sentence_after = sentences[int(m.group(2))].text

            sentence_text, sentence_offset = sentence.text, sentence.offset

            # print(sentence_text,
            #       sentence_text[start - sentence_offset:end - sentence_offset])

            if sentence_text[start - sentence_offset:end -
                             sentence_offset].lower() != text:
                print(text)

            sentence_text_left = sentence_before + " " + sentence_text[:start -
                                                                       sentence_offset]

            sentence_text_left_lists = re.split(' ', sentence_text_left)

            sentence_text_right = sentence_text[
                end - sentence_offset:] + " " + sentence_after

            sentence_text_right_lists = re.split(' ', sentence_text_right)

            context_partial = sentence_text_left_lists[
                -10:] + entity + sentence_text_right_lists[:10]

            context_partial_text = " ".join(context_partial)
            context_full = " ".join(sentence_text_left_lists + entity +
                                    sentence_text_right_lists)

            entity_text = " ".join(entity)
            print(context_partial_text, entity_text)

            input_partial.append(
                [m.group(6), "CUI", entity_text, context_partial_text])

            input_full.append(
                [m.group(6), "CUI", entity_text, context_full])

            # ann = {
            #     'start': start,
            #     'end': end,
            #     'type': m.group(6),
            #     'a': m.group(7),
            #     'text': text,
            #     'line': int(m.group(2)) - 1,
            #     'context'
            #     'id': f'{pathname.name}.l{i}'
            # }
            # print(1)

        # print(annotation)

    read.save_in_tsv(
        "data/i2b2_2010/semantic_group/input/context_parital_1.tsv",
        input_partial)

    read.save_in_tsv("data/i2b2_2010/semantic_group/input/context_full_1.tsv",
                     input_full)


generate_text_concept_input()
