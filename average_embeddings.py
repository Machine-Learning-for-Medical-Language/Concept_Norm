import read_files as read
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import os

def main(syn_path, cui_path, cui_idx_path, file_name):
    embeddings = np.load(syn_path)
    cuis = read.read_from_json(cui_path)
    cui_idx = read.read_from_json(cui_idx_path)
    avg = []
    for cui in cuis:
        s,e = cui_idx[cui]
        embedding_syn = embeddings[s:e]
        avg.append(np.mean(embedding_syn, axis = 0))
    avg = np.asarray(avg)
    
    read.create_folder(file_name)
    np.save(file_name, avg)
    
# def main(syn_path, cui_path, cui_idx_path, file_name):
#     embeddings = read.read_from_pickle(syn_path)
#     cuis = read.read_from_json(cui_path)
#     cui_idx = read.read_from_json(cui_idx_path)
#     avg = []
#     # for cui in cuis:
#     #     s,e = cui_idx[cui]
#     #     embedding_syn = embeddings[s:e]
#     #     avg.append(np.mean(embedding_syn, axis = 0))
#     avg = np.mean(embeddings, axis = 0)
#     read.save_in_pickle(file_name, avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentence embedding for each sentence in the sentence corpus ')

    parser.add_argument('--syn_path',
                        help='the direcotory of the model',required= True)

    parser.add_argument('--cui_path',
                        help='the type of the model, sentence_bert or just bert',required= True)

    parser.add_argument('--cui_idx_path',
                        help='the direcotory of the sentence corpus',required=True)

    parser.add_argument('--file_name',
                        help='the direcotory of the sentence corpus',required=True)

    args = parser.parse_args()
    syn_path = args.syn_path
    cui_path = args.cui_path
    cui_idx_path = args.cui_idx_path
    file_name = args.file_name
    main(syn_path, cui_path, cui_idx_path, file_name)
