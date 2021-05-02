import os
import argparse
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

import read_files as read
from CnlpBertConceptNorm import CnlpBertForClassification


def main(model_path, save_path):

    # model_path = "/home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/checkpoint_2/"

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels_list=[128, 434056],
        finetuning_task=["st_joint", "cn_joint"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        add_prefix_space=True,
        use_fast=True,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.save_pretrained(save_path)

    model = CnlpBertForClassification.from_pretrained(
        model_path,
        config=config,
        num_labels_list=[128, 434056],
        scale=45,
        margin=0.35,
        cache_dir=None,
        layer=-1,
        tokens=False,
        freeze=False,
        tagger=[False],
        concept_embeddings_pre=False,
        st_parameters_pre=False)
    model.bert.save_pretrained(save_path)
    np.save(os.path.join(save_path, "classfication_weights"),
            model.classifier.out_proj.weight.data)

    np.save(os.path.join(save_path, "classfication_bias"),
            model.classifier.out_proj.bias.data)

    np.savetxt(os.path.join(save_path, "threshold.txt"),
               [model.cosine_similarity.threshold.detach().numpy()])

    # BERT = AutoModel.from_pretrained("data/bert")
    # weights = model.cosine_similarity.weight.detach().numpy()
    # threshold = model.cosine_similarity.threshold.detach().numpy()
    # print(threshold)
    t = np.loadtxt(os.path.join(save_path, "threshold.txt"))
    print(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--model_path',
                        help='the direcotory of the model',
                        required=True)

    parser.add_argument(
        '--save_path',
        help='the type of the model, sentence_bert or just bert',
        required=True)
    
    args = parser.parse_args()
    model_path = args.model_path
    save_path = args.save_path
    
    main(model_path, save_path)


