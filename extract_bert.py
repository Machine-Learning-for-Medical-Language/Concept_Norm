from transformers import AutoConfig, AutoModel, AutoTokenizer

import read_files as read
from CnlpBertConceptNorm import CnlpBertForClassification


def load_model():

    model_path = "/home/dongfangxu/Projects/Concept_Norm/data/n2c2/models/checkpoint_2/"

    config = AutoConfig.from_pretrained(model_path,
                                        num_labels_list=[434056],
                                        finetuning_task="st_joint")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        add_prefix_space=True,
        use_fast=True,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.save_pretrained("data/bert_2")

    model = CnlpBertForClassification.from_pretrained(
        model_path,
        config=config,
        num_labels_list=[434056],
        scale=45,
        margin=0.35,
        cache_dir=None,
        layer=-1,
        tokens=False,
        freeze=False,
        tagger=[False],
        concept_embeddings_pre=True)
    model.bert.save_pretrained("data/bert_2")

    # BERT = AutoModel.from_pretrained("data/bert")
    # weights = model.cosine_similarity.weight.detach().numpy()
    threshold = model.cosine_similarity.threshold.detach().numpy()
    print(threshold)


load_model()
