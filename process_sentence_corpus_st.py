import argparse

import numpy as np
from sentence_transformers import SentenceTransformer, models

import read_files as read
from Pooling_custom import Pooling as Pooling
from transformer_custom import Transformer


def main(model_path, model_type, sentence_corpus, output_path):

    #### Read sentence courpus.  output: list of sentences ####
    sentences = read.read_from_tsv(sentence_corpus)
    sentences = [item for row in sentences for item in row]
    print(sentences[:10])

    if model_type.lower() in ["bert"]:
        # Load pretrained model
        word_embedding_model = Transformer(model_path)

        pooling_model = Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_mark_tokens=True)

        # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=2048, activation_function=nn.Tanh())

        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

        #### load sentence BERT models and generate sentence embeddings ####
    else:
        #### load sentence BERT models and generate sentence embeddings ####
        embedder = SentenceTransformer(model_path)

    embedder.max_seq_length = 64
    sentences_embedding = embedder.encode(sentences,
                                          batch_size=1024,
                                          show_progress_bar=True,
                                          num_workers=8)

    read.create_folder(output_path)

    np.save(output_path, sentences_embedding)

    # pickle.dump(d, open("file", 'w'), protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--model',
                        help='the direcotory of the model',
                        required=True)

    parser.add_argument(
        '--model_type',
        help='the type of the model, sentence_bert or just bert',
        required=True)

    parser.add_argument('--sentences',
                        help='the direcotory of the sentence corpus',
                        required=True)

    parser.add_argument('--output',
                        help='the direcotory of the sentence corpus',
                        required=True)




    main(model_path, model_type, sentence_corpus, output_path)
