import logging
import math
import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertConfig, BertForSequenceClassification, BertModel, \
    BertPreTrainedModel

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertaConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

# ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "roberta-base",
#     "roberta-large",
#     "roberta-large-mnli",
#     "distilroberta-base",
#     "roberta-base-openai-detector",
#     "roberta-large-openai-detector",
#     # See all RoBERTa models at https://huggingface.co/models?filter=roberta
# ]


class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, features, **kwargs):
        pass


class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, *kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tranform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, *kwargs):
        x = self.dropout(features)
        x = self.tranform(features)
        x = self.activation(x)
        return x


class RepresentationProjectionLayer(nn.Module):
    def __init__(self,
                 config,
                 layer=-1,
                 tokens=False,
                 tagger=False,
                 mean=False):
        super().__init__()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()
        self.layer_to_use = layer
        self.tokens = tokens
        self.mean = mean
        self.tagger = tagger
        self.hidden_size = config.hidden_size
        if tokens and tagger:
            raise Exception(
                'Inconsistent configuration: tokens and tagger cannot both be true'
            )

    def forward(self, features, event_tokens, input_features, **kwargs):
        seq_length = features[0].shape[1]
        if self.tokens:
            # grab the average over the tokens of the thing we want to classify
            # probably involved passing in some sub-sequence of interest so we know what tokens to grab,
            # then we average across those tokens.
            token_lens = event_tokens.sum(1)
            expanded_tokens = event_tokens.unsqueeze(2).expand(
                features[0].shape[0], seq_length, self.hidden_size)
            filtered_features = features[self.layer_to_use] * expanded_tokens
            x = filtered_features.sum(1) / token_lens.unsqueeze(1).expand(
                features[0].shape[0], self.hidden_size)
        elif self.tagger:
            x = features[self.layer_to_use]
        elif self.mean:
            token_ids = input_features['input_ids']
            attention_mask = input_features['attention_mask']
            meaningful_token_ids = [
                i.index(3) for i in token_ids.cpu().numpy().tolist()
            ]
            for i in meaningful_token_ids:
                attention_mask[:, i] = 0
                attention_mask[:, i + 1:] = 0
            token_embeddings = features[self.layer_to_use]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded,
                                       1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            # take <s> token (equiv. to [CLS])
            x = features[self.layer_to_use][:, 0, :]
            # x = self.dropout(x)
            # x = self.dense(x)
            # x = self.activation(x)
        return x


import torch.nn.functional as F
from torch.nn import Parameter


class CosineLayer(nn.Module):
    def __init__(self,
                 concept_dim=(434056, 768),
                 concept_embeddings_pre=False,
                 path=None):
        super(CosineLayer, self).__init__()

        if concept_embeddings_pre:
            weights_matrix = np.load(
                os.path.join(path,
                             "concept_embeddings.npy")).astype(np.float32)
            self.weight = Parameter(torch.from_numpy(weights_matrix),
                                    requires_grad=False)
            threshold_value = np.loadtxt(os.path.join(
                path, "threshold.txt")).astype(np.float32)

            self.threshold = Parameter(torch.tensor(threshold_value),
                                       requires_grad=False)
        else:

            self.weight = Parameter(torch.rand(concept_dim),
                                    requires_grad=True)
            self.threshold = Parameter(torch.tensor(0.45), requires_grad=True)

    def forward(self, features):
        eps = 1e-8
        batch_size, fea_size = features.shape
        input_norm, weight_norm = features.norm(
            2, dim=1, keepdim=True), self.weight.norm(2, dim=1, keepdim=True)
        input_norm = torch.div(
            features, torch.max(input_norm, eps * torch.ones_like(input_norm)))
        weight_norm = torch.div(
            self.weight,
            torch.max(weight_norm, eps * torch.ones_like(weight_norm)))
        sim_mt = torch.mm(input_norm, weight_norm.transpose(0, 1))

        cui_less_score = torch.full((batch_size, 1), 1).to(
            features.device) * self.threshold.to(features.device)
        similarity_score = torch.cat((sim_mt, cui_less_score), 1)
        return similarity_score


class ArcMarginProduct(nn.Module):
    def __init__(self, s, m, easy_margin=True):
        super(ArcMarginProduct, self).__init__()

        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine_score, labels):
        sine = torch.sqrt((1.0 - torch.pow(cosine_score, 2)).clamp(0, 1))
        phi = cosine_score * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine_score > 0, phi, cosine_score)
        else:
            phi = torch.where(cosine_score > self.th, phi,
                              cosine_score - self.mm)

        one_hot = torch.zeros(cosine_score.size()).to(cosine_score.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_score)
        output *= self.s
        return output


class CnlpBertForClassification(nn.Module):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(
            self,
            model_name="",
            config=None,
            num_labels_list=[16, 434056],
            # num_labels_list=[434056],
            mu1=1,
            mu2=0,
            scale=20,
            margin=0.5,
            layer=-1,
            freeze=False,
            tokens=True,
            tagger=[False, False],
            concept_embeddings_pre=False):

        super(CnlpBertForClassification, self).__init__()
        self.num_labels = num_labels_list
        self.mu1 = mu1
        self.mu2 = mu2
        self.name_or_path = model_name

        self.bert = BertModel.from_pretrained(self.name_or_path)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        # self.bert.init_weights()

        self.feature_extractor_mention = RepresentationProjectionLayer(
            config, layer=layer, tokens=True, tagger=tagger[0])

        self.feature_extractor_context = RepresentationProjectionLayer(
            config, layer=layer, tokens=True, tagger=tagger[0], mean=False)

        # self.lstm = nn.LSTM(768 * 2,
        #                     768,
        #                     1,
        #                     batch_first=True,
        #                     bidirectional=True)

        concept_st = np.load("data/umls/cui_sg_matrix.npy").astype(np.float32)
        self.concept_2_st = torch.from_numpy(concept_st)

        self.st_transformation = torch.nn.MaxPool1d(kernel_size=434057,
                                                    return_indices=True)

        # self.mlp = Mlp(config)

        # self.normalize = torch.nn.Softmax(dim=1)

        if len(self.num_labels) > 1:

            self.classifier = ClassificationHead(config, self.num_labels[0])

        self.arcface = ArcMarginProduct(s=scale, m=margin, easy_margin=True)

        self.bert_mention = BertModel.from_pretrained(self.name_or_path)
        for param in self.bert_mention.parameters():
            param.requires_grad = False
        # self.bert_mention.init_weights()

        self.cosine_similarity = CosineLayer(
            concept_dim=(434056, 768),
            concept_embeddings_pre=concept_embeddings_pre,
            path=self.name_or_path)

        self.prob = nn.Softmax(dim=1)

        self.class_weights = torch.tensor([
            333.3125, 1.5150568181818183, 0.4300806451612903,
            0.5088740458015267, 3.9213235294117648, 0.17570506062203478, 1, 1,
            23.808035714285715, 23.808035714285715, 1, 1, 11.904017857142858,
            6.944010416666667, 0.2264351222826087, 2.6880040322580645
        ])

        # Are we operating as a sconcepts_presentation
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        input_ids_c=None,
        attention_mask_c=None,
        token_type_ids_c=None,
        position_ids_c=None,
        head_mask=None,
        inputs_embeds=None,
        event_tokens=None,
        event_tokens_c=None,
        st_labels=None,
        concept_labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        labels = [st_labels, concept_labels]
        outputs = self.bert_mention(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=True,
                                    return_dict=True)

        outputs_context = self.bert(input_ids_c,
                                    attention_mask=attention_mask_c,
                                    token_type_ids=token_type_ids_c,
                                    position_ids=position_ids_c,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=True,
                                    return_dict=True)

        batch_size, seq_len = input_ids.shape

        mu = [self.mu1, self.mu2]

        logits = []

        loss = 0

        features_mention = self.feature_extractor_mention(
            outputs.hidden_states, event_tokens, None)

        features_context = self.feature_extractor_context(
            outputs_context.hidden_states, event_tokens_c, None)

        cui_logits_intermediate = self.cosine_similarity(features_mention)

        st_logits_intermediate = cui_logits_intermediate.unsqueeze(
            1) * self.concept_2_st.T.to(cui_logits_intermediate.device)

        st_logits, cui_indexs = self.st_transformation(st_logits_intermediate)

        st_logits = st_logits.squeeze(-1)
        cui_indexs = cui_indexs.squeeze(-1)

        st_logits_values, st_logits_index = torch.topk(st_logits, 3)

        st_logits_multihot = torch.sum(torch.eye(
            self.num_labels[0])[st_logits_index],
                                       dim=1)
        st_logits_multihot = st_logits_multihot.to(st_logits.device)

        context_logits = self.classifier(features_context)
        # context_logits = self.prob(context_logits)
        # context_score = 0.9* st_logits + 0.1 *context_logits
        context_score = st_logits_multihot * context_logits

        if self.bert.training:
            cui_logits_output = self.arcface(cui_logits_intermediate,
                                             labels[1])
            task_logits = cui_logits_output

        else:
            task_logits = cui_logits_intermediate

        logits = [context_score, task_logits]

        for task_ind, task_num_labels in enumerate(self.num_labels):
            # if task_ind == 0 and len(self.num_labels) == 2:
            #     # task_logits = self.classifier(features_mention)
            #     task_logits_st_intermediate = self.cosine_similarity_st(
            #         features_mention)
            #     st_logits.append(task_logits_st_intermediate)
            #     if self.training:

            #         task_logits_st_output = self.arcface(
            #             task_logits_st_intermediate, labels[task_ind])
            #         task_logits = task_logits_st_output

            #     else:
            #         task_logits = task_logits_st_intermediate

            # logits.append(task_logits)

            if labels[task_ind] is not None:
                # if task_ind == 0:
                #     loss_fct = CrossEntropyLoss(weight=class_weights)
                # else:
                loss_fct = CrossEntropyLoss()
                # task_logits_new = task_logits.view(-1,
                #                                    self.num_labels[task_ind])
                labels_new = labels[task_ind].view(-1)

                task_loss = loss_fct(logits[task_ind], labels_new)

                if loss is None:
                    loss = task_loss
                else:
                    loss += mu[task_ind] * task_loss

        if self.bert.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs_context.hidden_states,
                attentions=outputs_context.attentions,
            )
        else:
            logits = [context_score, cui_indexs]
            return SequenceClassifierOutput(loss=loss, logits=logits)
