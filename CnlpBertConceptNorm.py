import logging
import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
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


class RepresentationProjectionLayer(nn.Module):
    def __init__(self, config, layer=-1, tokens=False, tagger=False):
        super().__init__()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()
        self.layer_to_use = layer
        self.tokens = tokens
        self.tagger = tagger
        self.hidden_size = config.hidden_size
        if tokens and tagger:
            raise Exception(
                'Inconsistent configuration: tokens and tagger cannot both be true'
            )

    def forward(self, features, event_tokens, **kwargs):
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
                 concept_embeddings_pre=False):
        super(CosineLayer, self).__init__()

        if concept_embeddings_pre == True:
            weights_matrix = np.load(
                "data/n2c2/triplet_network/con_norm/ontology+train+dev_con_embeddings.npy"
            )
            self.weight = Parameter(torch.from_numpy(weights_matrix),
                                    requires_grad=True)
        else:
            self.weight = Parameter(torch.rand(concept_dim),
                                    requires_grad=True)

        self.threshold = Parameter(torch.tensor(0.502746), requires_grad=True)

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

        cui_less_score = torch.full(
            (batch_size, 1), 1).to(features.device) * self.threshold
        similarity_score = torch.cat((sim_mt, cui_less_score), 1)
        return similarity_score


class ArcMarginProduct(nn.Module):
    def __init__(self, s, m, easy_margin=False):
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
        # if self.easy_margin:
        #     phi = torch.where(cosine_score > 0, phi, cosine_score)
        # else:
        #     phi = torch.where(cosine_score > self.th, phi,
        #                       cosine_score - self.mm)

        one_hot = torch.zeros(cosine_score.size()).to(cosine_score.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_score)
        output *= self.s
        return output


class CnlpBertForClassification(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(
        self,
        config,
        num_labels_list=[1],
        scale=20,
        margin=0.5,
        layer=-1,
        freeze=False,
        tokens=True,
        tagger=[False],
        concept_embeddings_pre=False,
    ):

        super().__init__(config)
        self.num_labels = num_labels_list

        self.bert = BertModel(config)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = True

        # weights_matrix = np.load(
        #     "data/n2c2/triplet_network/st_subpool/ontology+train+dev_con_embeddings.npy"
        #     # "data/n2c2/triplet_network/mark_subpool/ontology_con_embeddings.npy"
        # )

        # self.linear = nn.Embedding(num_embeddings, embedding_dim)

        self.cosine_similarity = CosineLayer(
            concept_dim=(434056, 768),
            concept_embeddings_pre=concept_embeddings_pre)

        self.feature_extractor = RepresentationProjectionLayer(
            config, layer=layer, tokens=tokens, tagger=tagger[0])

        # self.feature_extractor_mention = RepresentationProjectionLayer(
        #     config, layer=layer, tokens=tokens, tagger=tagger[0])

        # self.classifier = ClassificationHead(config, self.num_labels[0])

        # for task_ind, task_num_labels in enumerate(self.num_labels):
        #     self.classifiers.append(ClassificationHead(config,
        #                                                task_num_labels))

        self.arcface = ArcMarginProduct(s=scale, m=margin, easy_margin=True)

        self.init_weights()

        # Are we operating as a sconcepts_presentation
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        # input_ids_m=None,
        # attention_mask_m=None,
        # token_type_ids_m=None,
        # position_ids_m=None,
        head_mask=None,
        inputs_embeds=None,
        event_tokens=None,
        # event_tokens_m=None,
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
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=True,
                            return_dict=True)

        # outputs_mention = self.bert(input_ids_m,
        #                             attention_mask=attention_mask_m,
        #                             token_type_ids=token_type_ids_m,
        #                             position_ids=position_ids_m,
        #                             head_mask=head_mask,
        #                             inputs_embeds=inputs_embeds,
        #                             output_attentions=output_attentions,
        #                             output_hidden_states=True,
        #                             return_dict=True)

        batch_size, seq_len = input_ids.shape

        logits = []

        loss = None

        features = self.feature_extractor(outputs.hidden_states, event_tokens)
        # features_mention = self.feature_extractor_mention(
        #     outputs_mention.hidden_states, event_tokens_m)
        # features = 0.5 * features + 0.5 *features_mention

        for task_ind, task_num_labels in enumerate(self.num_labels):
            # if task_ind == 1:
            #     task_logits = self.classifier(features)
            # else:
            task_logits_intermediate = self.cosine_similarity(features)

            if self.training:
                task_logits_output = self.arcface(task_logits_intermediate,
                                                  labels[task_ind])
                task_logits = task_logits_output

            else:
                task_logits = task_logits_intermediate

            logits.append(task_logits)

            if labels[task_ind] is not None:
                loss_fct = CrossEntropyLoss()
                # task_logits_new = task_logits.view(-1,
                #                                    self.num_labels[task_ind])
                labels_new = labels[task_ind].view(-1)

                task_loss = loss_fct(task_logits, labels_new)

                if loss is None:
                    loss = task_loss
                else:
                    loss += task_loss

        #### semantic type classifier ####

        # if labels is not None:
        #     loss_fct1 = CrossEntropyLoss()
        # task_loss1 = loss_fct1(task_logits1.view(-1, self.num_labels[0]),
        #                        st_labels.view(-1))
        # if loss is None:
        #     loss = task_loss1
        # else:
        #     loss += task_loss1

        ### Concept representation layer ####

        # task_logits = self.cosine_similarity(features)

        # if concept_labels is not None:

        # loss_fct2 = CosineEmbeddingLoss()
        # task_loss2 = loss_fct2(
        #     features, concepts_presentation,
        #     torch.Tensor(features.size(0)).cuda().fill_(1.0))
        # if loss is None:
        #     loss = task_loss2
        # else:
        #     loss += task_loss2

        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=loss, logits=logits)
