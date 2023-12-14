#  ReGCL: Rethinking Message Passing in Graph Contrastive Learning
Code for ReGCL model.

## Dependencies

- Python 3.8
- PyTorch 1.13.1+ cu117 
- torch-geometric 2.3.0       
- torch-scatter 2.1.0
- torch-sparse 0.6.15
- torch-spline-conv 1.2.1        
- pyyaml 6.0.1
- scikit-learn 1.3.0
- numpy 1.21.6                                                                         


## Datasets

Citation Networks: 'Cora', 'Citeseer' and 'Pubmed'.

Co-occurence Networks: 'Amazon-Photo', 'Coauthor-CS' 

| Dataset      | # Nodes | # Edges | # Classes | # Features |
| ------------ | ------- | ------- | --------- | ---------- |
| Cora         | 2,708   | 10,556  | 7         | 1,433      |
| Citeseer     | 3,327   | 9,228   | 6         | 3,703      |
| Pubmed       | 19,717  | 88,651  | 3         | 500        |
| Amazon-Photo | 7,650   | 287,326 | 8         | 745        |
| Coauthor-CS  | 18,333  | 327,576 | 15        | 6,805      |

## Usage
To run the codes, use the following commands:
```python
#test:
python train.py --dataset Cora  --test 
#train:
python train.py --dataset Cora  --lr 5e-4 --tau 0.2 --dfr1 0.4 --dfr2 0.4 --der1 0.0 --der2 0.4
```

