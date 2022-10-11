# PromCSE: Improved Universal Sentence Embeddings with Prompt-based Contrastive Learning and Energy-based Learning

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sick)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sick?p=deep-continuous-prompt-for-contrastive-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts12)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts12?p=deep-continuous-prompt-for-contrastive-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts13)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts13?p=deep-continuous-prompt-for-contrastive-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts14)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts14?p=deep-continuous-prompt-for-contrastive-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts16)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts16?p=deep-continuous-prompt-for-contrastive-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts15)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts15?p=deep-continuous-prompt-for-contrastive-1)

This repository contains the code for our EMNLP 2022 paper [Improved Universal Sentence Embeddings with Prompt-based Contrastive Learning and Energy-based Learning](https://arxiv.org/abs/2203.06875v2). Our code is modified based on [SimCSE](https://github.com/princeton-nlp/SimCSE) and [P-tuning v2](https://github.com/THUDM/P-tuning-v2/). Here we would like to sincerely thank them for their excellent works.

We release our best model checkpoint which acquires **Top 1** results on four STS tasks:

<!-- <img src="https://github.com/YJiangcm/DCPCSE/blob/master/figure/leaderboard.png" width="700" height="380"> -->

|          Model          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg. |
|:-----------------------:|:-----:|:----------:|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  sup-PromCSE-RoBERTa-large [download](https://drive.google.com/drive/folders/1OWqNgsPtzvsxmDDnOf0WaVuNkELEcatG?usp=sharing)  |  79.14 |88.64| 83.73| 87.33 |84.57| 87.84| 82.07| 84.76|
|  unsup-PromCSE-BERT-base [download](https://drive.google.com/drive/folders/1OcgJ-7gU_N7J7x5ezrigFLlTU8h7Uvjx?usp=sharing)  |  73.03 |85.18| 76.70| 84.19 |79.69| 80.62| 70.00| 78.49|

If you have any questions, feel free to raise an issue.

[//]: <## Architecture>
[//]: <We add multi-layer trainable dense vectors as soft prompts to the input sequence, which means the input embeddings as well as each layer's hidden embeddings of prompts are optimized (the orange blocks). Note that all parameters of the pre-trained model are frozen (the blue blocks), thus reducing the number of tunable parameters to around **0.1\%**. The [CLS] token embedding of the last layer is selected as the sentence representation. The contrastive framework is the same as SimCSE.>


## Setups
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

Run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

## Training

**Data**

Following SimCSE, we use the same datasets to train our unsupervised models and supervised models. You can run `data/download_wiki.sh` and `data/download_nli.sh` to download the two datasets.

**Training scripts**  
(The same as `run_unsup_example.sh`)
```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-promcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --learning_rate 3e-2 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --pre_seq_len 16 \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16
 ```

We provide example training scripts for both unsupervised and supervised PromCSE. In `run_unsup_example.sh`, we provide a single-GPU (or CPU) example for the unsupervised version, and in `run_sup_example.sh` we give a **multiple-GPU** example for the supervised version. Both scripts call `train.py` for training. We explain the arguments in following:
* `--train_file`: Training file path. We support "txt" files (one line for one sentence) and "csv" files (2-column: pair data with no hard negative; 3-column: pair data with one corresponding hard negative instance). You can use our provided Wikipedia or NLI data, or you can use your own data with the same format.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: We have found that for unsupervised PromCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised PromCSE models.
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.
* `--pre_seq_len`: The length of deep continuous prompt.
* `--prefix_projection`: Whether apply a two-layer MLP head over the prompt embeddings.
* `--prefix_hidden_size`: The hidden size of the MLP projection head if prefix_projection is used.
* `--do_eh_loss`: Whether to use Energy-based Hinge loss in supervised models. If True:
  * `--eh_loss_margin`: Margin of Energy-based Hinge loss.
  * `--eh_loss_weight`: Weight of Energy-based Hinge loss.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to evaluate the model on the STS-B development set (need to download the dataset following the [evaluation](#evaluation) section) and save the best checkpoint.

All our experiments are conducted on Nvidia 3090 GPUs.

**Hyperparameters**

| **Unsupervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 64       | 64
| Learning rate  | 3e-2 | 3e-2 | 3e-2 | 1e-2 |
| Prompt length | 16 | 10 | 14 | 10 |
| do_mlm | False | False | True | True |
| Epoch |1|1|1|1|
| Valid steps | 125 | 125 | 125 | 125 |

    
| **Supervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 512       | 512
| Learning rate  | 1e-2 | 5e-3 | 1e-2 | 5e-3 |
| Prompt length | 12 | 12 | 10 | 10 |
| do_mlm | False | False | False | False |
| Epoch |10|10|10|10|
| Valid steps | 125 | 125 | 125 | 125 |


## Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. The STS tasks include seven standard STS tasks (STS12-16, STSB, SICK-R) and one domain-shifted STS task (CxC).

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```
To evaluate the domain shift robustness of sentence embedding, we need to download [CxC](https://drive.google.com/drive/folders/1ZnRlVlc4kFsKbaWj9cFbb8bQU0fxzz1c?usp=sharing), and put the data into *SentEval/data/downstream/CocoCXC*

Then come back to the root directory, you can evaluate the well trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path result/my-sup-promcse-roberta-large \
    --pooler_type cls \
    --task_set sts \
    --mode test \
    --pre_seq_len 10
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 79.14 | 88.64 | 83.73 | 87.33 | 84.57 |    87.84     |      82.07      | 84.76 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. 
* `--pooler_type`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised PromCSE**, you should use this option.
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised PromCSE**, you should take this option.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `cococxc`: Evaluate on domain-shifted CXC task.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.
* `--pre_seq_len`: The length of deep continuous prompt.

## Usage
We provide *tool.py* to easily compute the cosine similarities between two groups of sentences as well as build index for a group of sentences and search among them. You can have a try by runing
```bash
python tool.py \
    --model_name_or_path result/my-unsup-promcse-bert-base-uncased \
    --pooler_type cls_before_pooler \
    --pre_seq_len 16
```

which is expected to output the following results.
```
=========Calculate cosine similarities between queries and sentences============

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.18it/s]100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 42.26it/s][[0.5904227  0.70516586 0.65185255 0.82756    0.6969594  0.85966974
  0.58715546 0.8467339  0.6583321  0.6792214 ]
 [0.6125869  0.73508096 0.61479807 0.6182762  0.6161849  0.59476817
  0.595963   0.61386335 0.694822   0.938746  ]]

=========Naive brute force search============

2022-10-09 11:59:06,004 : Encoding embeddings for sentences...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.03it/s]2022-10-09 11:59:06,029 : Building index...
2022-10-09 11:59:06,029 : Finished
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 95.40it/s]100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 115.25it/s]Retrieval results for query: A man is playing music.
    A man plays the piano.  (cosine similarity: 0.8597)
    A man plays a guitar.  (cosine similarity: 0.8467)
    A man plays the violin.  (cosine similarity: 0.8276)
    A woman is reading.  (cosine similarity: 0.7051)
    A man is eating food.  (cosine similarity: 0.6969)
    A woman is taking a picture.  (cosine similarity: 0.6792)
    A woman is slicing a meat.  (cosine similarity: 0.6583)
    A man is lifting weights in a garage.  (cosine similarity: 0.6518)

Retrieval results for query: A woman is making a photo.
    A woman is taking a picture.  (cosine similarity: 0.9387)
    A woman is reading.  (cosine similarity: 0.7351)
    A woman is slicing a meat.  (cosine similarity: 0.6948)
    A man plays the violin.  (cosine similarity: 0.6183)
    A man is eating food.  (cosine similarity: 0.6162)
    A man is lifting weights in a garage.  (cosine similarity: 0.6148)
    A man plays a guitar.  (cosine similarity: 0.6139)
    An animal is biting a persons finger.  (cosine similarity: 0.6126)


=========Search with Faiss backend============

2022-10-09 11:59:06,055 : Loading faiss with AVX2 support.
2022-10-09 11:59:06,092 : Successfully loaded faiss with AVX2 support.
2022-10-09 11:59:06,093 : Encoding embeddings for sentences...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.17it/s]2022-10-09 11:59:06,335 : Building index...
2022-10-09 11:59:06,335 : Use GPU-version faiss
2022-10-09 11:59:06,447 : Finished
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 101.44it/s]Retrieval results for query: A man is playing music.
    A man plays the piano.  (cosine similarity: 0.8597)
    A man plays a guitar.  (cosine similarity: 0.8467)
    A man plays the violin.  (cosine similarity: 0.8276)
    A woman is reading.  (cosine similarity: 0.7052)
    A man is eating food.  (cosine similarity: 0.6970)
    A woman is taking a picture.  (cosine similarity: 0.6792)
    A woman is slicing a meat.  (cosine similarity: 0.6583)
    A man is lifting weights in a garage.  (cosine similarity: 0.6519)

Retrieval results for query: A woman is making a photo.
    A woman is taking a picture.  (cosine similarity: 0.9387)
    A woman is reading.  (cosine similarity: 0.7351)
    A woman is slicing a meat.  (cosine similarity: 0.6948)
    A man plays the violin.  (cosine similarity: 0.6183)
    A man is eating food.  (cosine similarity: 0.6162)
    A man is lifting weights in a garage.  (cosine similarity: 0.6148)
    A man plays a guitar.  (cosine similarity: 0.6139)
    An animal is biting a persons finger.  (cosine similarity: 0.6126)
```

## Citation

Please cite our paper by:

```bibtex
@misc{jiang2022promcse,
      title={Improved Universal Sentence Embeddings with Prompt-based Contrastive Learning and Energy-based Learning}, 
      author={Yuxin Jiang, Linhan Zhang and Wei Wang},
      year={2022},
      eprint={2203.06875},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
