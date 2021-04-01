import re

import read_files as read


def load_document(file):
    text = ''
    sent_ids = []
    sent_offsets = []
    sentence = []
    with open(file, 'r') as f:
        offset = 0
        for index, line in enumerate(f):
            text += line
            sent_offsets.append(offset)
            offset += len(line)
            sent_ids.extend([index] * len(line))
    return text, sent_ids, sent_offsets


def load_concept(file_path):
    concept_list = []
    concept_span_list = []
    with open(file_path, 'r') as f:

        for l in f:
            line = l.strip().split('||')
            concept_id = line[0]
            concept = line[1]
            concept_pos = [int(item) for item in line[2:]]
            concept_spans = []
            for span_idx in range(int(len(concept_pos) / 2)):
                concept_span_single = (concept_pos[span_idx * 2],
                                       concept_pos[span_idx * 2 + 1])
                concept_spans.append(concept_span_single)
                concept_span_list.append(concept_span_single)

            concept_list.append([concept_id, concept, concept_spans])
    # concept_list_new = sorted(concept_list, key=lambda x: x[1][0])

    return concept_list


def match_token_span(text, tokens, spans, concepts):

    if len(concepts) == 0:
        return tokens, spans
    else:
        concept_spans = []
        for concept in concepts:
            concept_spans += concept[2]
        concept_spans = list(set(concept_spans))
        concept_span_sorted = sorted(concept_spans, key=lambda x: x[0])
        tokens_new = []
        spans_new = []

    # exceptions = [
    #     "Head magnetic resonance imaging study and magnetic resonance imaging angiogram with and without gadolinium .\n",
    #     "No cervical , para-auricular or clavicular lymphadenopathy .\n",
    #     "The patient was placed on suicide precautions , and was hydrated with an infusion of intravenous fluid .\n",
    #     # "The tears in the transverse mesocolon and colon serosa were repaired .\n",
    #     "Cardiac catheterization demonstrated elevated right and left-sided filling pressures at rest , with pulmonary hypertension and normal cardiac index .\n",
    #     "took Prednisone 40 at home and presented to ED where found to have 1mm lateral ST depressions and slight troponin and MB leak .\n",
    #     "In 10/92 , she had a CT scan which showed fatty infiltration of her liver diffusely with a 1 cm cyst in the right lobe of the liver .\n",
    #     ""
    # ]
    # if text in exceptions:
    #     print(1)

    for idx, token_span in enumerate(spans):
        # if idx == 6:
        #     print(1)
        token_info = {}
        for concept_span in concept_span_sorted:
            if concept_span[0] > token_span[0] and token_span[
                    1] <= concept_span[1] and token_span[1] > concept_span[0]:
                token_info = read.add_dict(token_info, str(idx),
                                           (concept_span[0], token_span[1]))
            elif token_span[0] >= concept_span[0] and token_span[
                    1] > concept_span[1] and token_span[0] < concept_span[1]:
                token_info = read.add_dict(token_info, str(idx),
                                           (token_span[0], concept_span[1]))

            elif token_span[0] < concept_span[0] and token_span[
                    1] > concept_span[1]:
                token_info = read.add_dict(token_info, str(idx),
                                           (concept_span[0], concept_span[1]))
            elif token_span[0] >= concept_span[0] and token_span[
                    1] <= concept_span[1]:
                token_info = read.add_dict(token_info, str(idx),
                                           (token_span[0], token_span[1]))

        if str(idx) not in token_info:
            tokens_new.append(tokens[idx])
            spans_new.append(token_span)

        else:
            token_info_spans = token_info[str(idx)]
            subtoken_span = list(token_span)
            for token_info_span in token_info_spans:
                subtoken_span.append(token_info_span[0])
                subtoken_span.append(token_info_span[1])
            subtoken_span = sorted(list(set(subtoken_span)))
            for idx, _ in enumerate(subtoken_span[:-1]):
                tokens_new.append(text[subtoken_span[idx]:subtoken_span[idx +
                                                                        1]])
                spans_new.append((subtoken_span[idx], subtoken_span[idx + 1]))
    return tokens_new, spans_new


def create_tagging_discontinuous(text, tokens, token_spans, concepts,
                                 note_name):
    note_name = note_name.replace(".txt", "")
    print(text, tokens, token_spans, concepts)
    print()
    # exceptions = [
    #     "Head magnetic resonance imaging study and magnetic resonance imaging angiogram with and without gadolinium .\n",
    #     "No cervical , para-auricular or clavicular lymphadenopathy .\n",
    #     "The patient was placed on suicide precautions , and was hydrated with an infusion of intravenous fluid .\n",
    #     "The tears in the transverse mesocolon and colon serosa were repaired .\n",
    #     "Cardiac catheterization demonstrated elevated right and left-sided filling pressures at rest , with pulmonary hypertension and normal cardiac index .\n",
    #     "took Prednisone 40 at home and presented to ED where found to have 1mm lateral ST depressions and slight troponin and MB leak .\n",
    #     "In 10/92 , she had a CT scan which showed fatty infiltration of her liver diffusely with a 1 cm cyst in the right lobe of the liver .\n",
    #     ""
    # ]
    # if text in exceptions:
    #     print()
    concepts = sorted(concepts, key=lambda x: len(x[2]))
    for concept in concepts:
        concept_id = concept[0]
        # if concept_id == "N103":
        #     print(0)
        concept_cui = concept[1]
        concept_span = concept[2]
        concept_text = concept[3]
        # if len(concept_span) > 1:
        concept_span_new = []
        concept_text_new = []
        for span_single, text_single in zip(concept_span, concept_text):

            if " " in text_single:
                text_spans = [(m.start(), m.end())
                              for m in re.finditer(r'\S+', text_single)]
                concept_text_split = [
                    text_single[text_span[0]:text_span[1]]
                    for text_span in text_spans
                ]
                text_spans = [(item[0] + span_single[0],
                               item[1] + span_single[0])
                              for item in text_spans]
                concept_span_new += text_spans
                concept_text_new += concept_text_split
            elif text_single == "left-sided" and span_single == (56, 66):
                concept_text_new += ["left-", 'sided']
                concept_span_new += [(56, 61), (61, 66)]

            else:
                concept_span_new += [span_single]
                concept_text_new += [text_single]

        index_0 = token_spans.index(concept_span_new[0])
        index_last = token_spans.index(concept_span_new[-1])
        if not isinstance(tokens[index_0], list):
            tokens[index_0] = [concept_id, concept_cui, concept_text_new]
            for concept_span_new_single in concept_span_new[1:]:
                index_k = token_spans.index(concept_span_new_single)
                if not isinstance(tokens[index_k], list):
                    tokens[index_k] = ""

        elif isinstance(tokens[index_0],
                        list) and not isinstance(tokens[index_last], list):
            for concept_span_new_single in concept_span_new[1:-1]:
                index_k = token_spans.index(concept_span_new_single)

                if not isinstance(tokens[index_k], list):
                    tokens[index_k] = ""
            tokens[index_last] = [concept_id, concept_cui, concept_text_new]
        elif len(concept_span_new) >= 3:
            token_to_concept = False
            for concept_span_new_single in concept_span_new[1:-1]:
                index_k = token_spans.index(concept_span_new_single)
                if not isinstance(tokens[index_k],
                                  list) and token_to_concept is True:
                    tokens[index_k] = ""
                else:
                    tokens[index_k] = [
                        concept_id, concept_cui, concept_text_new
                    ]
                    token_to_concept = True
        else:
            tokens.insert(index_last,
                          [concept_id, concept_cui, concept_text_new])
    concept_list = [item for item in tokens if isinstance(item, list)]

    if len(concept_list) != len(concepts):
        raise ValueError("Number of tokens != Number of spans")

    print(tokens)
    print()
    print()
    token_new = []
    tag_new = []
    for token in tokens:
        if isinstance(token, list):
            concept_id = token[0]
            concept_cui = token[1]
            concept_tokens = token[2]
            token_new += concept_tokens
            tag_new.append("B_" + concept_id + "_" + concept_cui + "_" +
                           " ".join(concept_tokens) + "+++" + note_name)
            for concept_token in concept_tokens[1:]:
                tag_new.append("I_" + concept_id + "_" + concept_cui + "_" +
                               " ".join(concept_tokens) + "+++" + note_name)
        elif len(token) > 0:
            token_new += [token]
            tag_new.append("O")

    return token_new, tag_new


# def create_tagging(text, tokens, token_spans, concepts, note_name):
#     n_tokens = len(tokens)
#     note_name = note_name.replace(".txt", "")

#     if len(tokens) != len(token_spans):
#         raise ValueError("Number of tokens != Number of spans")

#     tags = ["O"] * n_tokens
#     for concept in concepts:
#         concept_text = concept[0]
#         concept_cui = concept[1]

#         concept_id = concept[0]
#         concept_span = concept[2][0]metrics_eval
#         concept_text = concept_id + "_" + concept_cui + "_" + " ".join(
#             concept[3])
#         tagged = False
#         for idx, span in enumerate(token_spans):
#             if span[0] == concept_span[0] and span[1] <= concept_span[1]:
#                 tags[idx] = "B_" + concept_text + "+++" + note_name
#                 tagged = True
#             elif span[0] > concept_span[0] and span[1] <= concept_span[1]:
#                 tags[idx] = "I_" + concept_text + "+++" + note_name
#                 tagged = True
#         if tagged is False:
#             raise ValueError("Number of tokens != Number of spans")

#     # print(text, tokens, tags)
#     # print(concepts)
#     # print()
#     return tokens, tags


def get_st_cui(semantic_type, cui):

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


def raw_accuracy_score(gold_labels, pre_labels):
    count = len(gold_labels)
    label_in = 0
    for gold_label, pre_label in zip(gold_labels ,pre_labels):
        if gold_label in pre_label:
            label_in +=1
    print(label_in/count)
