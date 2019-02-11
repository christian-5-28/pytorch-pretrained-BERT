# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import time

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from wrapper_sentence_classifier import WrapperClassifier
from wrapper_sentence_classifier import BertFeatExtractor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


class SemEvalTrainer:

    def __init__(self, args):
        self.args = args
        self.initialize_run()

    def initialize_run(self):

        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                device = self.args.gpu
                cudnn.benchmark = True
                cudnn.enabled = True
                # torch.cuda.manual_seed_all(args.evaluation_seed)
        else:
            print('CUDA NOT AVAILABLE!')
            time.sleep(20)

        layer_indexes = [int(x) for x in self.args.layers.split(",")]

        # PREPARING TRAIN AND EVAL DATA #
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)

        self.train_sentences, self.train_labels = self.read_examples(self.args.train_input_file)
        self.train_features = self.convert_examples_to_features(
            examples=self.train_sentences, seq_length=self.args.max_seq_length, tokenizer=self.tokenizer)

        self.eval_sentences, self.eval_labels = self.read_examples(self.args.eval_input_file)
        self.eval_features = self.convert_examples_to_features(
            examples=self.eval_sentences, seq_length=self.args.max_seq_length, tokenizer=self.tokenizer)

        # CREATING THE TRAIN AND EVAL DATA LOADERS
        self.train_dataloader = self.create_data_loader(self.train_features, self.train_labels)
        self.eval_dataloader = self.create_data_loader(self.eval_features, self.eval_labels)

        print('data loaders ready...')

        # INITIALIZING THE MODEL #
        feat_extractor = BertFeatExtractor(args=self.args)

        classifier_head = torch.nn.LSTM(input_size=self.args.input_size,
                                        hidden_size=self.args.hidden_size,
                                        batch_first=True,
                                        bidirectional=self.args.bidirectional)

        self.model = WrapperClassifier(feat_extractor=feat_extractor,
                                       classifier_head=classifier_head,
                                       freeze_feat_extract=True)

        # gpu model handling
        if self.args.cuda:
            if self.args.single_gpu:
                logger.info('USING SINGLE GPU!')
                self.model = self.model.cuda()
            else:
                self.model = torch.nn.DataParallel(self.model, dim=1).cuda()

        # initializing the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)

    def train(self):

        ce_loss = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(self.args.epochs):

            self.model.train()
            tr_loss = 0
            for batch_id, (input_ids, input_mask, example_indices, labels) in self.train_dataloader:
                input_ids = input_ids.to(self.args.gpu)
                input_mask = input_mask.to(self.args.gpu)

                logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)

                # loss = ce_loss(logits.view(-1, self.args.num_labels), labels.view(-1))
                loss = ce_loss(logits, labels)
                tr_loss += loss.item()

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch_id % self.args.log_interval == 0 and batch_id > 0:
                    cur_loss = tr_loss / self.args.log_interval

                    elapsed = time.time() - start_time

                    logger.info('| epoch {:3d} | {:5d} batch | lr {:02.2f} | ms/batch {:5.2f} | '
                                     'loss {:5.2f} '.format(
                        epoch, batch_id, self.optimizer.param_groups[0]['lr'],
                                      elapsed * 1000 / self.args.log_interval, cur_loss))

                    tr_loss = 0
                    start_time = time.time()

            self.evaluate_model()





    def create_data_loader(self, features, labels):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_labels = torch.tensor([lable for lable in labels], dtype=torch.long)

        data = TensorDataset(all_input_ids,
                             all_input_mask,
                             all_example_index,
                             all_labels
                             )

        if self.args.local_rank == -1:
            sampler = SequentialSampler(data)
        else:
            sampler = DistributedSampler(data)

        dataloader = DataLoader(data, sampler=sampler, batch_size=self.args.batch_size)

        return dataloader







def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default='data/train/raw_semEval_bert_emoji_conv.txt', type=str)
    parser.add_argument("--output_file", default='data/train/out_feat_bert_train.txt', type=str)
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--cuda",
                        action='store_true',
                        default=True)
    parser.add_argument("--single_gpu",
                        action='store_true',
                        default=True)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.set_device(args.gpu)
            device = args.gpu
            cudnn.benchmark = True
            cudnn.enabled = True
            # torch.cuda.manual_seed_all(args.evaluation_seed)
    else:
        print('CUDA NOT AVAILABLE!')
        time.sleep(20)

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    examples = read_examples(args.input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    print('features ready...')

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    # model.to(device)

    # gpu handling
    if args.cuda:
        if args.single_gpu:
            logger.info('USING SINGLE GPU!')
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model, dim=1).cuda()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    start_extract = time.time()
    sentences_time = []
    with open(args.output_file, "w", encoding='utf-8') as writer:

        for input_ids, input_mask, example_indices in enumerate(eval_dataloader):
            start_batch = time.time()
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            # forward trough the Bert model. It returns the hidden outputs of all its layers
            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            # print('size of the encoder layers: {}'.format(all_encoder_layers[0].size()))

            all_encoder_layers = all_encoder_layers

            '''
            for b, example_index in enumerate(example_indices):

                start_sentence = time.time()

                # taking the specific preprocessed sample object
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)

                # feature = unique_id_to_feature[unique_id]

                output_json = collections.OrderedDict()

                # saving the sentence unique id
                output_json["linex_index"] = unique_id

                layer_outputs = [all_encoder_layers[int(layer_index)].detach().cpu().numpy() for layer_index in
                                 layer_indexes]

                all_out_features = []
                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        # taking the layer output of the chosen layer,
                        # detaching it from the graph and convert it to numpy array
                        # detach_start = time.time()
                        # layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_outputs[j]

                        # print('detach_time: {} seconds'.format(time.time() - detach_start))
                        # selecting the output of the specific sentence
                        # (this is due because we are using a batch size > 1)
                        layer_output = layer_output[b]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index

                        # 'i' is used to select the vector related to the ith token
                        layers["values"] = [
                            round(x.item(), 6) for x in layer_output[i]
                        ]
                        all_layers.append(layers)

                    out_features = collections.OrderedDict()
                    out_features["token"] = token
                    out_features["layers"] = all_layers
                    all_out_features.append(out_features)

                # features representation for all the tokens in the sentence
                output_json["features"] = all_out_features

                end_sentence = (time.time() - start_sentence, len(feature.tokens))
                sentences_time.append(end_sentence)

                # writer.write(json.dumps(output_json) + "\n")
            '''
            print(len(all_encoder_layers))
            print(all_encoder_layers[0].shape)
            all_encoder_layers = torch.stack(all_encoder_layers, dim=0)
            print('old shape: {}'.format(all_encoder_layers.shape))
            all_encoder_layers = all_encoder_layers.permute(1, 2, 0, 3)
            print('permute shape: {}'.format(all_encoder_layers.shape))
            all_encoder_layers = all_encoder_layers.contiguous().view(all_encoder_layers.shape[0], all_encoder_layers.shape[1], -1)
            print('new shape: {}'.format(all_encoder_layers.shape))

            print('end batch, {} sec'.format(time.time() - start_batch))




    print('extraction ended, time: {:5.2f} seconds \n'.format((time.time() - start_extract)))

    avg_tok = 0
    avg_time = 0
    for i, (sent_time, tokens) in enumerate(sentences_time):
        print(
            'extraction sentence {} with {} tokenks, sentence time: {:5.2f} seconds, token time: {:5.4f} seconds'.format(
                i, tokens, sent_time, sent_time / tokens))
        avg_tok += tokens
        avg_time += sent_time

    avg_time_tok = avg_time / avg_tok
    avg_tok = avg_tok / len(sentences_time)
    avg_time = avg_time / len(sentences_time)
    print('\n On average there are {:5.2f} tokens per sentence'.format(avg_tok))
    print('\n On average time elapsed per sentence: {:5.4f} seconds'.format(avg_time))
    print('\n On average time elapsed per token: {:5.4f} seconds'.format(avg_time_tok))
    # print('\n elapsed time: {:5.6f} seconds'.format(time.time() - start_time))


if __name__ == "__main__":
    start = time.time()
    main()
    print('features extraction concluded in {} seconds'.format(time.time() - start))
