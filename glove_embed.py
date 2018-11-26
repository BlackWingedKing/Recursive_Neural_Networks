import torch
import torchnlp
from torchnlp.word_to_vector import GloVe
import nltk
from nltk.corpus import treebank


# load the data 
# data_list is a list of file names for data 
# ext -> .mrg gt ext -> .prd
data_ext = '.mrg'
gt_ext = '.prd'

data_list = treebank.fileids()
# print the data returns list of strings... for every sentences
print(treebank.words('wsj_0003.mrg'))
# print the tree
# print(treebank.parsed_sents('wsj_0003.mrg')[0])
# print the gt of each word
# print(treebank.tagged_words('wsj_0003.mrg'))


vectors = GloVe()
vectors['hello']
# here vectors has the all embedding vectors of the words
