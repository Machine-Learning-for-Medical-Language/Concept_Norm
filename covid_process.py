import os

import spacy
from medspacy.custom_tokenizer import create_medspacy_tokenizer
from medspacy.sentence_splitting import PyRuSHSentencizer

import read_files as read


def tokenization(file_dir, output_dir):
    nlp = spacy.blank("en")
    medspacy_tokenizer = create_medspacy_tokenizer(nlp)
    default_tokenizer = nlp.tokenizer

    sentencizer = PyRuSHSentencizer(rules_path="./resources/rush_rules.tsv")

    nlp.add_pipe(sentencizer)
    covid_input = []

    notes_dir = ['note1.txt', 'note2.txt', 'note3.txt']
    for note_dir in notes_dir:
        print(note_dir)
        texts = read.textfile2list(os.path.join(file_dir, note_dir))
        for text in texts:
            text = text.strip()
            if len(text) > 0:
                doc = nlp(text)
                for sent in doc.sents:
                    sent_text = sent.text
                    print(sent_text)
                    # sent_text_default_tokenizer = [
                    #     item.text
                    #     for item in list(default_tokenizer(sent_text))
                    # ]
                    # print("Tokens from default tokenizer:",
                    #       sent_text_default_tokenizer)

                    sent_text_medspacy_tokenizer = [
                        item.text.strip()
                        for item in list(medspacy_tokenizer(sent_text))
                        if len(item.text.strip()) > 0
                    ]
                    sent_text = ' '.join(sent_text_medspacy_tokenizer)
                    labels = ' '.join(['O'] *
                                      len(sent_text_medspacy_tokenizer))
                    covid_input.append([labels, sent_text])
                    print("Tokens from medspacy tokenizer:",
                          sent_text_medspacy_tokenizer)
        covid_input.append(['O', 'EOS'])

        read.save_in_tsv(os.path.join(output_dir, 'test.tsv'), covid_input)


tokenization('data/covid_data/notes/', 'data/covid_data/input/')
