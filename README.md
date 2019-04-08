# Paragraph Ranker
This is a PyTorch implementation of a paper [Ranking Paragraphs for Improving Answer Recall in Open-Domain Question Answering (Lee et al. 2018)](https://arxiv.org/abs/1810.00494). Code implementations are based on the [DrQA repository](https://github.com/facebookresearch/DrQA), and a few additional experiments on entity based QA were conducted with [@donghyeonk](https://github.com/donghyeonk).

## Requirements
- Install [cuda-9.0](https://developer.nvidia.com/cuda-90-download-archive)
- Install [Pytorch 0.4.1](https://pytorch.org/)
- Python version >= 3.5 is required

## Setups
Follow datasets and enviromental setups described in the [DrQA repository](https://github.com/facebookresearch/DrQA).

## Preprocessings
ParagraphRanker uses datasets preprocessed with SimpleTokenizer of DrQA.
```bash
# Use preprocess.py to preprocess with SimpleTokenizer
$ python scripts/reader/preprocess.py data/datasets/ datasets/ --tokenizer simple
$ python scripts/reader/preprocess.py data/datasets/ datasets/ --tokenizer simple --split SQuAD-v1.1-dev
```

## Pretraining and Evaluation
```bash
# Pretrain reader model
$ python entityqa/reader/train.py

# Pretrain ranker model
$ python entityqa/ranker/train.py

# Evaluate QA pipeline
$ python entityqa/pipeline/predict.py --query-type SQuAD --ranker-type default --reader-type default
```
To run each train.py or predict.py, you should modify the file directories defined in the first few lines of each file. See codes for argument details of each file.
