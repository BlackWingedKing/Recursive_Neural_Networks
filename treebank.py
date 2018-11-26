import torch
from torchnlp.datasets import penn_treebank_dataset
train = penn_treebank_dataset(train=True)
print(train[:100])