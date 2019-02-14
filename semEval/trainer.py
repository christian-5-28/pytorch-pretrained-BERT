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
import os
import pickle
import re
import sys
import time
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from wrapper_sentence_classifier import WrapperClassifier
from wrapper_sentence_classifier import BertFeatExtractor
from tensorboardX import SummaryWriter

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


class SemEvalTrainer:

    def __init__(self, args):
        self.args = args
        self.initialize_run()

    @staticmethod
    def read_examples(input_file, targets_path):
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

        with open(targets_path, 'rb') as file:
            targets_list = pickle.load(file)

        return examples, targets_list

    @staticmethod
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

    def convert_examples_to_features(self, examples, seq_length, tokenizer):
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
                self._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
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
                self.logger.info("*** Example ***")
                self.logger.info("unique_id: %s" % (example.unique_id))
                self.logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features

    def initialize_run(self):
        self.search_dir = 'bert_semEval_layers_{}_{}'.format(self.args.layers, time.strftime("%Y%m%d-%H%M%S"))

        # creating the tensorboard directory
        if not os.path.exists(self.args.tboard_path):
            os.mkdir(self.args.tboard_path)

        tboard_path = os.path.join(self.args.tboard_path, self.search_dir)
        self.writer = SummaryWriter(tboard_path)

        self.search_dir = os.path.join(self.args.main_path, self.search_dir)

        if not os.path.exists(self.search_dir):
            os.mkdir(self.search_dir)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.search_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger("bert_semEval")
        self.logger.addHandler(fh)

        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.seed)
        else:
            print('CUDA NOT AVAILABLE!')
            time.sleep(20)

        # PREPARING TRAIN AND EVAL DATA #
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)

        self.train_sentences, self.train_labels = self.read_examples(self.args.train_input_file,
                                                                     targets_path=self.args.train_targets_path)
        self.train_features = self.convert_examples_to_features(
            examples=self.train_sentences, seq_length=self.args.max_seq_length, tokenizer=self.tokenizer)

        self.eval_sentences, self.eval_labels = self.read_examples(self.args.eval_input_file,
                                                                   targets_path=self.args.eval_targets_path)
        self.eval_features = self.convert_examples_to_features(
            examples=self.eval_sentences, seq_length=self.args.max_seq_length, tokenizer=self.tokenizer)

        # CREATING THE TRAIN AND EVAL DATA LOADERS
        self.train_dataloader = self.create_data_loader(self.train_features, self.train_labels)
        self.eval_dataloader = self.create_data_loader(self.eval_features, self.eval_labels)

        print('data loaders ready...')

        # INITIALIZING THE MODEL #
        feat_extractor = BertFeatExtractor(args=self.args)
        num_layers = len([int(x) for x in self.args.layers.split("_")])
        classifier_head = torch.nn.LSTM(input_size=num_layers * self.args.bert_hidden_size,
                                        hidden_size=self.args.rnn_hidden_size,
                                        batch_first=True,
                                        num_layers=self.args.rnn_layers,
                                        bidirectional=self.args.bidirectional)

        self.model = WrapperClassifier(feat_extractor=feat_extractor,
                                       sentence_head=classifier_head,
                                       args=self.args)

        # gpu model handling
        if self.args.cuda:
            if self.args.single_gpu:
                self.logger.info('USING SINGLE GPU!')
                self.model = self.model.cuda()
            else:
                self.model = torch.nn.DataParallel(self.model, dim=1).cuda()

        # initializing the optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)

    def get_metrics(self, predictions, ground):
        """Given predicted labels and the respective ground truth labels, display some metrics
        Input: shape [# of samples, NUM_CLASSES]
            predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
            ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
        Output:
            accuracy : Average accuracy
            microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
            microRecall : Recall calculated on a micro level
            microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
        """
        # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
        label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
        emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
        print('shape pred: {}'.format(predictions.shape))
        print('shape ground: {}'.format(ground.shape))

        discretePredictions = np.zeros_like(predictions)
        argmax_indices = predictions.max(axis=1,keepdims=1) == predictions
        discretePredictions[argmax_indices] = 1

        zeros_ground = np.zeros((ground.shape[0], self.args.num_labels))
        zeros_ground[np.arange(ground.shape[0]), ground] = 1
        ground = zeros_ground



        truePositives = np.sum(discretePredictions * ground, axis=0)
        falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
        falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

        print("True Positives per class : ", truePositives)
        print("False Positives per class : ", falsePositives)
        print("False Negatives per class : ", falseNegatives)

        # ------------- Macro level calculation ---------------
        macroPrecision = 0
        macroRecall = 0
        # We ignore the "Others" class during the calculation of Precision, Recall and F1
        for c in range(1, self.args.num_labels):
            precision = truePositives[c] / (truePositives[c] + falsePositives[c])
            macroPrecision += precision
            recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
            macroRecall += recall
            f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
            print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

        macroPrecision /= 3
        macroRecall /= 3
        macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (macroPrecision + macroRecall) > 0 else 0
        print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
        macroPrecision, macroRecall, macroF1))

        # ------------- Micro level calculation ---------------
        truePositives = truePositives[1:].sum()
        falsePositives = falsePositives[1:].sum()
        falseNegatives = falseNegatives[1:].sum()

        print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (
        truePositives, falsePositives, falseNegatives))

        microPrecision = truePositives / (truePositives + falsePositives)
        microRecall = truePositives / (truePositives + falseNegatives)

        microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (microPrecision + microRecall) > 0 else 0
        # -----------------------------------------------------

        predictions = predictions.argmax(axis=1)
        ground = ground.argmax(axis=1)
        accuracy = np.mean(predictions == ground)

        return accuracy, microPrecision, microRecall, microF1

    def evaluate_model(self):

        self.model.eval()
        eval_loss = 0
        num_batches = 1
        predictions = []
        gt = []
        for batch_id, (input_ids, input_mask, example_indices, labels) in enumerate(self.eval_dataloader):
            input_ids = input_ids.to(self.args.gpu)
            input_mask = input_mask.to(self.args.gpu)
            labels = labels.to(self.args.gpu)

            logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)

            # loss = ce_loss(logits.view(-1, self.args.num_labels), labels.view(-1))
            loss = self.ce_loss(logits.view(-1, self.args.num_labels), labels.view(-1))
            eval_loss += loss.item()
            num_batches += 1
            log_prob = torch.nn.functional.log_softmax(logits.view(-1, self.args.num_labels), dim=-1)
            predictions.append(log_prob.detach().cpu())
            gt.append(labels.view(-1).detach().cpu())

        avg_val_loss = eval_loss / num_batches
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.detach().cpu().numpy()

        gt = torch.cat(gt, dim=0)
        gt = gt.detach().cpu().numpy()

        accuracy, microPrecision, microRecall, microF1 = self.get_metrics(predictions, gt)

        self.logger.info("VALIDATION METRICS:| Loss: {:.4f} |  Accuracy : {:.4f} |  Micro Precision : {:.4f}|  Micro "
                         "Recall : {:.4f} |  Micro F1 : {:.4f} |".format(
        avg_val_loss, accuracy, microPrecision, microRecall, microF1))

        return avg_val_loss, accuracy, microPrecision, microRecall, microF1

    def train(self):

        self.ce_loss = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(self.args.epochs):

            self.model.train()
            tr_loss = 0
            for batch_id, (input_ids, input_mask, example_indices, labels) in enumerate(self.train_dataloader):
                input_ids = input_ids.to(self.args.gpu)
                input_mask = input_mask.to(self.args.gpu)
                labels = labels.to(self.args.gpu)

                logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)
                loss = self.ce_loss(logits.view(-1, self.args.num_labels), labels.view(-1))
                # loss = self.ce_loss(logits, labels)
                tr_loss += loss.item()

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch_id % self.args.log_interval == 0 and batch_id > 0:
                    cur_loss = tr_loss / self.args.log_interval

                    elapsed = time.time() - start_time

                    self.logger.info('| epoch {:3d} | {:5d} batch | lr {:02.4f} | ms/batch {:5.2f} | '
                                     'loss {:5.2f} '.format(
                        epoch + 1, batch_id, self.optimizer.param_groups[0]['lr'],
                                      elapsed * 1000 / self.args.log_interval, cur_loss))

                    tr_loss = 0
                    start_time = time.time()

            self.writer.add_scalar('train_loss', cur_loss, epoch)

            avg_val_loss, accuracy, microPrecision, microRecall, microF1 = self.evaluate_model()

            self.writer.add_scalar('valid_loss', avg_val_loss, epoch)
            self.writer.add_scalar('valid_accuracy', accuracy, epoch)
            self.writer.add_scalar('valid_microPrecision', microPrecision, epoch)
            self.writer.add_scalar('valid_microRecall', microRecall, epoch)
            self.writer.add_scalar('valid_microF1', microF1, epoch)

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

