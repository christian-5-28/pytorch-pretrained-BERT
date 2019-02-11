import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class WrapperClassifier(nn.Module):

    def __init__(self, feat_extractor, sentence_head, args):

        super(WrapperClassifier, self).__init__()

        self.args = args
        self.feat_extractor = feat_extractor
        self.sentence_head = sentence_head
        self.classifier = nn.Linear(self.args.hidden_size, self.args.num_labels)

        if self.args.freeze_feat_extract:
            for child in self.feat_extractor.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, input_x):

        input_x = self.feat_extractor(input_x)
        output, (last_hidden, last_memory) = self.sentence_head(input_x)

        logits = self.classifier(last_hidden)

        return logits


class BertFeatExtractor(nn.Module):

    def __init__(self, args):

        super(BertFeatExtractor, self).__init__()
        self.feat_extractor = BertModel.from_pretrained(self.args.bert_model)
        self.feat_layers  = [int(x) for x in args.layers.split(",")]

    def forward(self, input_ids, token_type_ids, attention_mask):

        all_encoder_layers, _ = self.feat_extractor(input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)

        # taking the only layers desired
        all_encoder_layers = [all_encoder_layers[int(layer_index)] for layer_index in self.feat_layers]

        # print(len(all_encoder_layers))
        # print(all_encoder_layers[0].shape)
        all_encoder_layers = torch.stack(all_encoder_layers, dim=0)

        # print('old shape: {}'.format(all_encoder_layers.shape))
        all_encoder_layers = all_encoder_layers.permute(1, 2, 0, 3)

        # changing the shape to concat the desired layers.
        # From (bs, seq_length, hidden_dim, num_layers) to --> [bs, seq_length, (hidden_dim * num_layers) ]

        # print('permute shape: {}'.format(all_encoder_layers.shape))
        all_encoder_layers = all_encoder_layers.contiguous().view(all_encoder_layers.shape[0],
                                                                  all_encoder_layers.shape[1], -1)
        # print('new shape: {}'.format(all_encoder_layers.shape))

        return all_encoder_layers

