import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: Lowercase the input
    """
    def __init__(self,
                 model_name_or_path: str,
                 start: bool = True,
                 max_seq_length: int = 128,
                 model_args: Dict = {},
                 cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {},
                 do_lower_case: Optional[bool] = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path,
                                            **model_args,
                                            cache_dir=cache_dir)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path,
                                                    config=config,
                                                    cache_dir=cache_dir)
        if start ==True:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                additional_special_tokens=['<e>', '</e>'],
                **tokenizer_args)
    
            self.auto_model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                # additional_special_tokens=['<e>', '</e>'],
                **tokenizer_args)
    
            # self.auto_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            'input_ids': features['input_ids'],
            'attention_mask': features['attention_mask']
        }
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({
            'token_embeddings': output_tokens,
            'cls_token_embeddings': cls_tokens,
            'attention_mask': features['attention_mask']
        })

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(
                    output_states
            ) < 3:  #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        mark_start_ind = self.tokenizer.convert_tokens_to_ids('<e>')
        mark_end_ind = self.tokenizer.convert_tokens_to_ids('</e>')
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        batch_output = self.tokenizer(*to_tokenize,
                                      padding=True,
                                      truncation='longest_first',
                                      return_tensors="pt",
                                      max_length=self.max_seq_length)
        mark_token_ids = self.generate_mark_tokens_id(
            batch_output["input_ids"], mark_start_ind, mark_end_ind)
        dtype = torch.long if isinstance(mark_token_ids[0][0],
                                         int) else torch.float
        mark_token_ids = torch.tensor(mark_token_ids, dtype=dtype)
        output.update(batch_output)
        output.update({'mark_token_ids': mark_token_ids})

        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'),
                  'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
                'sentence_bert_config.json', 'sentence_roberta_config.json',
                'sentence_distilbert_config.json',
                'sentence_camembert_config.json',
                'sentence_albert_config.json',
                'sentence_xlm-roberta_config.json',
                'sentence_xlnet_config.json'
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)

    @staticmethod
    def generate_mark_tokens_id(input_ids, mark_start_ind, mark_end_ind):
        masking_token_ids = []
        for input_id in input_ids:
            # inputs_mention = {
            #     k + "_m": batch_encoding_mentions[k][i]
            #     for k in batch_encoding_mentions
            # }
            # inputs.update(inputs_mention)
            input_id = input_id.cpu().numpy().tolist()
            try:
                event_start = input_id.index(mark_start_ind)
            except:
                event_start = -1

            try:
                event_end = input_id.index(mark_end_ind)
            except:
                event_end = len(input_id) - 1

            masking_token_id = [0] * len(input_id)
            if event_start >= 0:
                masking_token_id = [0] * event_start + [1] * (
                    event_end - event_start + 1) + [0] * (len(input_id) -
                                                          event_end - 1)
            else:
                masking_token_id = [1] * len(input_id)
            masking_token_ids.append(masking_token_id)
        return masking_token_ids
