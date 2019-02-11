import argparse

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
                    help="local_rank for distributed training on gpus")
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

