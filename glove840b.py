import bcolz
import numpy as np
import pickle
words = []
idx = 0
word2idx = {}
glove_path = '/home/ritesh/Desktop/ML_project/glove.840B/'
vectors = bcolz.carray(np.zeros(1), rootdir=glove_path + 'glove.840B.300.dat', mode='w')

with open(glove_path + 'glove.840B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        print(idx)
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((2200000, 300)), rootdir=glove_path + 'glove.840B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(glove_path + 'glove.840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(glove_path + 'glove.840B.300_idx.pkl', 'wb'))