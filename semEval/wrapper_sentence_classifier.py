import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class WrapperClassifier(nn.Module):

    def __init__(self, feat_extractor, sentence_head, args):

        super(WrapperClassifier, self).__init__()

        self.args = args
        self.feat_extractor = feat_extractor
        self.sentence_head = sentence_head

        self.dropout = torch.nn.Dropout(p=self.args.dropout)

        hidden_size = self.args.rnn_hidden_size if not self.args.bidirectional and not self.args.rnn_layers > 1 else 2 * self.args.rnn_hidden_size
        self.classifier = nn.Linear(hidden_size, self.args.num_labels)

        '''
        if self.args.freeze_feat_extract:
            for child in self.feat_extractor.children():
                for param in child.parameters():
                    param.requires_grad = False
            
        '''

    def forward(self, input_ids, token_type_ids, attention_mask):

        input_x = self.feat_extractor(input_ids, token_type_ids, attention_mask)
        input_x = self.dropout(input_x)
        output, (last_hidden, last_memory) = self.sentence_head(input_x)

        if self.args.bidirectional or self.args.rnn_layers > 1:
            last_hidden = last_hidden.permute(1, 0, 2)
            last_hidden = last_hidden.contiguous().view(last_hidden.shape[0], -1)

        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)

        return logits


class BertFeatExtractor(nn.Module):

    def __init__(self, args):

        super(BertFeatExtractor, self).__init__()
        self.args = args
        self.feat_extractor = BertModel.from_pretrained(self.args.bert_model)

        self.feat_layers = [int(x) for x in args.layers.split("_")]

        if self.args.use_combination_feat:
            self.feat_combiner = nn.Linear(len(self.feat_layers), self.args.bert_hidden_size)

        if not self.args.tuning_feat_extract:
            for child in self.feat_extractor.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):

        all_encoder_layers, _ = self.feat_extractor(input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)

        # taking the only layers desired
        all_encoder_layers = [all_encoder_layers[int(layer_index)] for layer_index in self.feat_layers]

        # print(len(all_encoder_layers))
        # print(all_encoder_layers[0].shape)
        all_encoder_layers = torch.stack(all_encoder_layers, dim=0)

        if self.args.use_combination_feat:
            all_encoder_layers = all_encoder_layers.permute(1, 2, 3, 0)
            all_encoder_layers = self.feat_combiner(all_encoder_layers)

        else:
            all_encoder_layers = all_encoder_layers.permute(1, 2, 0, 3)

            # removing representation of the [CLS] token
            all_encoder_layers = all_encoder_layers[:, 1:, :, :]

            # changing the shape to concat the desired layers.
            # From (bs, seq_length, hidden_dim, num_layers) to --> [bs, seq_length, (hidden_dim * num_layers) ]

            # print('permute shape: {}'.format(all_encoder_layers.shape))
            all_encoder_layers = all_encoder_layers.contiguous().view(all_encoder_layers.shape[0],
                                                                      all_encoder_layers.shape[1], -1)
            # print('new shape: {}'.format(all_encoder_layers.shape))

        return all_encoder_layers

