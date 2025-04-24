![FedICI logo](FedICI_logo.png)

# Integrating Commonality and Individuality for Graph Federated Learning: A Graph Spectrum Perspective

Official Code Repository for our paper:

Integrating Commonality and Individuality for Graph Federated Learning: A Graph Spectrum Perspective


## Highlights
- User-Friendly: Simple and intuitive to use.
- SOTA Performance: Outperforms the second-best by 3.28% on all heterophilic datasets.
- Rich Datasets: Covers 11 GFL datasets (6 homophilic, 5 heterophilic).

## Requirement
- Python 3.8.8
- PyTorch 1.12.0+cu113
- PyTorch Geometric 2.3.0

## Subgraph generation
Download from the Google Drive (https://drive.google.com/file/d/1PyqvR6yL43Om42fdsbKHj5WCgREvi3St/view?usp=sharing) and then unzip it.

Place the `datasets` folder in the same path as `README.md`


## Parameter description


- `dataset`: specify dataset to use
- `n_clients`: specify the number of clients
- `mode`: specify the 'disjoint' or 'overlapping' subgraph partitioning scenario
 

# Homophilic datasets
Follow command lines to run the experiments.
## Cora
### non-overlapping
```Python
$ python main.py --dataset Cora --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Cora --n_clients 10 --mode overlapping
```


## CiteSeer
### non-overlapping
```Python
$ python main.py --dataset CiteSeer --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset CiteSeer --n_clients 10 --mode overlapping
```


## PubMed
### non-overlapping
```Python
$ python main.py --dataset PubMed --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset PubMed --n_clients 10 --mode overlapping
```


## Amazon-Computer
### non-overlapping
```Python
$ python main.py --dataset Computers --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Computers --n_clients 10 --mode overlapping
```


## Amazon-Photo
### non-overlapping
```Python
$ python main.py --dataset Photo --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Photo --n_clients 10 --mode overlapping
```


## ogbn-arxiv
### non-overlapping
```Python
$ python main.py --dataset ogbn-arxiv --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset ogbn-arxiv --n_clients 10 --mode overlapping
```


# Heterophilic datasets

## Roman-empire
### non-overlapping
```Python
$ python main.py --dataset Roman-empire --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Roman-empire --n_clients 10 --mode overlapping
```


## Amazon-ratings
### non-overlapping
```Python
$ python main.py --dataset Amazon-ratings --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Amazon-ratings --n_clients 10 --mode overlapping
```


## Minesweeper
### non-overlapping
```Python
$ python main.py --dataset Minesweeper --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Minesweeper --n_clients 10 --mode overlapping
```


## Tolokers
### non-overlapping
```Python
$ python main.py --dataset Tolokers --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Tolokers --n_clients 10 --mode overlapping
```


## Questions
### non-overlapping
```Python
$ python main.py --dataset Questions --n_clients 10 --mode disjoint
```
### overlapping
```Python
$ python main.py --dataset Questions --n_clients 10 --mode overlapping
```
