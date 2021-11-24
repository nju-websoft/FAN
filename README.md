# Knowing False Negatives: An Adversarial Training Method for Distantly Supervised Relation Extraction

[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/GLRE/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/GLRE/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)

## Abstract
> Distantly supervised relation extraction (RE) automatically aligns unstructured
text with relation instances in a knowledge base (KB).
Due to the incompleteness of current KBs, sentences implying certain relations may
be annotated as N/A instances, which causes the so called false negative (FN) problem.
Current RE methods usually overlook this problem, inducing improper biases in both training and
testing procedures.
To address this issue, we propose a two-stage approach. First, it finds out possible
FN samples by heuristically leveraging the memory mechanism of deep neural networks.
Then, it aligns those unlabeled data with the training data into a unified feature
space by adversarial training to assign pseudo labels and further utilize the
information contained in them. Experiments on two wildly used benchmark datasets
demonstrate the effectiveness of our approach.

## File Structure
~~~text
FAN/
├─ bert/
├─ data/
    ├── gids/
    ├── nyt10/
├─ loader/
├─ mining/: code for Stage I: Mining
├─ model/: neural networks
├─ output/
├─ align.py
├─ align.sh
├─ utils.py
~~~

## Running
#### Stage I: Mining
~~~bash
$ cd mining
$ python run.py
$ python infer.py
~~~

#### Stage II: Aligning
~~~bash
python align.py --output_dir ./output --device cuda:0 --pos_file ./output/mining/nyt10/pos.txt
--unlabeled_file ./output/mining/nyt10/unlabeled.txt --rel2id ./data/nyt10/rel2id.json 
--epoch 10 --batch_size 160 --lr 1e-5 --pretrain ./bert/bert-base-uncased --optimizer sgd 
--contrastive --alpha 0.01 --beta 0.01 --gamma 0.0001 --tau 1.0 --weighting --max_bag_size -1
~~~

## Citation
~~~bibtex
@inproceedings{FAN,
 author = {Kailong Hao and Botao Yu and Wei Hu},
 title = {Knowing False Negatives: An Adversarial Training Method for Distantly Supervised Relation Extraction},
 booktitle = {EMNLP},
 year = {2021},
}
~~~


