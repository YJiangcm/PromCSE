
import logging
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer
from promcse.models import BertForCL, RobertaForCL
from tqdm import tqdm
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union
from torch import Tensor, device
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import io
import os
import random
import torch
from torch.autograd import Variable

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class PromCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by PromCSE.
    """
    def __init__(self, args, 
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        if 'roberta' in args.model_name_or_path:
            self.model = RobertaForCL.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=AutoConfig.from_pretrained(args.model_name_or_path),
                    cache_dir=args.cache_dir,
                    revision=args.model_revision,
                    use_auth_token=True if args.use_auth_token else None,
                    model_args=args
                )
        elif 'bert' in args.model_name_or_path:
            self.model = BertForCL.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=AutoConfig.from_pretrained(args.model_name_or_path),
                    cache_dir=args.cache_dir,
                    revision=args.model_revision,
                    use_auth_token=True if args.use_auth_token else None,
                    model_args=args
                )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings
    
    
    def similarity(self, queries: Union[str, List[str]], 
                    keys: Union[str, List[str], ndarray], 
                    device: str = None) -> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities
    
    
    def build_index(self, sentences_or_file_path: Union[str, List[str]], 
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True 
            except:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}
        
        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path))) 
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else: 
                logger.info("Use CPU-version faiss")

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def add_to_index(self, sentences_or_file_path: Union[str, List[str]],
                        device: str = None,
                        batch_size: int = 64):
        
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)
        
        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path
        logger.info("Finished")


    
    def search(self, queries: Union[str, List[str]], 
                device: str = None, 
                threshold: float = 0.6,
                top_k: int = 10) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results
            
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])
            

if __name__ == "__main__":
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
    

    example_sentences = [
        'An animal is biting a persons finger.',
        'A woman is reading.',
        'A man is lifting weights in a garage.',
        'A man plays the violin.',
        'A man is eating food.',
        'A man plays the piano.',
        'A panda is climbing.',
        'A man plays a guitar.',
        'A woman is slicing a meat.',
        'A woman is taking a picture.'
    ]
    example_queries = [
        'A man is playing music.',
        'A woman is making a photo.'
    ]

    model = PromCSE(args)

    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    similarities = model.similarity(example_queries, example_sentences)
    print(similarities)

    print("\n=========Naive brute force search============\n")
    model.build_index(example_sentences, use_faiss=False)
    results = model.search(example_queries)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")
    
    print("\n=========Search with Faiss backend============\n")
    model.build_index(example_sentences, use_faiss=True)
    results = model.search(example_queries)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")

