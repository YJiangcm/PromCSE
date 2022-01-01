# DCPCSE: Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings

This repository contains the code for our paper Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings. Our code is modified based on [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Architecture


## Training Details
All our experiments are conducted on two Nvidia 3090 GPUs.

| **Unsupervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 64       | 64
| Learning rate  | 3e-2 | 3e-2 | 3e-2 | 1e-2 |
| Prompt length | 16 | 10 | 14 | 10 |
| Muiti-task | False | False | True | True |
| Epoch |1|1|1|1|
| Valid steps | 125 | 125 | 125 | 125 |

    
| **Supervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 256       | 256
| Learning rate  | 5e-3 | 5e-3 | 1e-2 | 5e-3 |
| Prompt length | 12 | 12 | 10 | 10 |
| Muiti-task | False | False | False | False |
| Epoch |10|5|5|5|
| Valid steps | 125 | 125 | 125 | 125 |

## Citation

Please cite our paper by:

```bibtex
@inproceedings{
}
```
