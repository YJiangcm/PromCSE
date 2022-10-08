import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from promcse.models import RobertaForCL, BertForCL

import matplotlib.pyplot as plt, seaborn

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler_type", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--temp", type=float, 
            default=0.05, 
            help="Temperature for softmax.")
    parser.add_argument("--hard_negative_weight", type=float, 
            default=0.0, 
            help="The **logit** of weight for hard negatives (only effective if hard negatives are used).")
    parser.add_argument("--do_mlm", action='store_true', 
            help="Whether to use MLM auxiliary objective.")
    parser.add_argument("--mlm_weight", type=float, 
            default=0.1, 
            help="Weight for MLM auxiliary objective (only effective if --do_mlm).")
    parser.add_argument("--mlp_only_train", action='store_true', 
            help="Use MLP only during training")
    parser.add_argument("--pre_seq_len", type=int, 
            default=10, 
            help="The length of prompt")
    parser.add_argument("--prefix_projection", action='store_true', 
            help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", type=int, 
            default=512, 
            help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--do_eh_loss", 
            action='store_true',
            help="Whether to add Energy-based Hinge loss")
    parser.add_argument("--eh_loss_margin", type=float, 
            default=None, 
            help="The margin of Energy-based Hinge loss")
    parser.add_argument("--eh_loss_weight", type=float, 
            default=None, 
            help="The weight of Energy-based Hinge loss")
    parser.add_argument("--cache_dir", type=str, 
            default=None,
            help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--model_revision", type=str, 
            default="main",
            help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", action='store_true', 
            help="Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models).")
    
    
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na', 'cococxc'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")

    
    args = parser.parse_args()
    
    if args.do_eh_loss:
        if args.eh_loss_margin is None or args.eh_loss_weight is None:
            parser.error('Requiring eh_loss_margin and eh_loss_weight if do_eh_loss is provided')
    
    # Load transformers' model checkpoint
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if 'roberta' in args.model_name_or_path:
        model = RobertaForCL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
                revision=args.model_revision,
                use_auth_token=True if args.use_auth_token else None,
                model_args=args                  
            )
    elif 'bert' in args.model_name_or_path:
        model = BertForCL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
                revision=args.model_revision,
                use_auth_token=True if args.use_auth_token else None,
                model_args=args
            )

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'cococxc':
        args.tasks = ['CocoCXC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True, output_attentions=True) ###############################
            pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # # Apply different poolers
        # if args.pooler == 'cls':
        #     # There is a linear+activation layer after CLS representation
        #     return pooler_output.cpu()
        # elif args.pooler == 'cls_before_pooler':
        #     return last_hidden[:, 0].cpu()
        # elif args.pooler == "avg":
        #     return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        # elif args.pooler == "avg_first_last":
        #     first_hidden = hidden_states[0]
        #     last_hidden = hidden_states[-1]
        #     pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        #     return pooled_result.cpu()
        # elif args.pooler == "avg_top2":
        #     second_last_hidden = hidden_states[-2]
        #     last_hidden = hidden_states[-1]
        #     pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        #     return pooled_result.cpu()
        # else:
        #     raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
