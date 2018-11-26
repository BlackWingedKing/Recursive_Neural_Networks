import imp
import torch
import torchnlp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import bcolz
import nltk
import pandas as pd
from nltk.corpus import treebank
from nltk import tree, treetransforms
from copy import deepcopy
from nltk.draw.tree import draw_trees
from nltk.tree import Tree
from nltk.tree import ParentedTree
import time
import numpy as np

# no of classes
num_leaf_classes = 23
num_node_classes = 6

final_map = {'NP':0, 'VP':1, 'PP':2, 'ADJP':3, 'S':4, 'X':5, 'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':22,'JJ':6,'LS':7,'MD':8,'NN':9, 'PDT':10,
            'POS':11,'PRP':12,'PP$':13,'RB':14,'SYM':15,'TO':16,'UH':17,'VB':18,'WDT':19,'WP':20,'WRB':21}
# maps for gt and leaf classes
# wmap = {0:'CC',1:'CD',2:'DT',3:'EX',4:'FW',5:'IN',6:'JJ',7:'JJR',8:'JJS',
#             9:'LS',10:'MD',11:'NN',12:'NNS',13:'NNP',14:'NNPS',15:'PDT',
#             16:'POS',17:'PRP',18:'PP$',19:'RB',20:'RBR',21:'RBS',22:'RP',
#             23:'SYM',24:'TO',25:'UH',26:'VB',27:'VBD',28:'VBG',29:'VBN',
#             30:'VBP',31:'VBZ',32:'WDT',33:'WP',34:'WP$',35:'WRB',36:'-NONE-'}

# smap = {0:'ADJP',1:'ADVP',2:'NP',3:'PP',4:'S',5:'SBAR',6:'SBARQ',7:'SINV',
#             8:'SQ',9:'VP',10:'WHADVP',11:'WHNP',12:'WHPP',13:'X',14:'*',
#             15:'0',16:'T'}

wmap1 = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'LS':7,'MD':8,'NN':9, 'PDT':10,
            'POS':11,'PRP':12,'PP$':13,'RB':14,'SYM':15,'TO':16,'UH':17,'VB':18,'WDT':19,'WP':20,'WRB':21,'X':22}

# smap1 = {'ADJP':0,'ADVP':1,'NP':2,'PP':3,'S':4,'SBAR':5,'SBARQ':6,'SINV':7,
#             'SQ':8,'VP':9,'WHADVP':10,'WHNP':11,'WHPP':12,'X':13,'*':14,
#             '0':15,'T':16}

smap1 = {'NP':0, 'VP':1, 'PP':2, 'ADJP':3, 'S':4, 'X':5}

# priority = ['NP', 'VP', 'PP', 'ADJP', 'S', 'X', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'LS', 'MD', 'NN', 'PDT', 'POS', 'PRP', 'PP$', 'RB', 'SYM', 'TO', 'UH', 'VB', 'WDT', 'WP', 'WRB']
priority = ['NP', 'VP', 'PP', 'ADJP', 'S', 'X']
# Data Loading part 
# data_list is a list of file names for data 
# ext -> .mrg gt ext -> .prd
data_ext = '.mrg'
gt_ext = '.prd'
data_path = '/home/ritesh/Desktop/ML_project/ptb.csv'
data_list = treebank.fileids()
data_list.pop(55)
data_list.pop(118)
train_data_list = data_list[0:150]
val_data_list = data_list[150:]

# this function loads the embedding vectors
embedding = pd.read_csv(data_path)
embedding.set_index("text", drop=True, inplace=True)
embedding = pd.DataFrame(embedding).T
embedding = embedding.to_dict('list')

# model definition for syntactic parsing...
class NodeNet(nn.Module):
    def __init__(self):
        super(NodeNet, self).__init__()
        embed_dim = 300
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)
        self.fc4 = nn.Linear(300,num_node_classes)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        p = self.fc1(x1) + self.fc2(x2)
        p = self.tanh(p)
        s = self.fc3(p)
        node_prob = F.log_softmax(self.fc4(p), dim=0)
        return s,p,node_prob

# model for parts of speech determination
class LeafNet(nn.Module):
    def __init__(self):
        super(LeafNet, self).__init__()
        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, num_leaf_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        p = self.fc2(self.tanh(self.fc1(x)))        
        return F.log_softmax(p, dim=0)


def get_embed(x):
    try:
        a = embedding[x]
    except:
        a = embedding['<unk>']
        #print('unknown word.... using <unk>')
    return a

def calculate_score(x, embed_x, model):
    # input an (word,gt), embed list of length l and model 
    # output score and a list of length l-l and p
    s = []
    p = []
    gt = []
    embed_list1 = embed_x
    for i in range (0,len(embed_x)-1):
        ts,tp,g = model(embed_x[i],embed_x[i+1])
        g = torch.argmax(g)
        s.append(ts)
        p.append(tp)
        gt.append(g)
    maxpos = s.index(max(s))
    embed_list1[maxpos] = p[maxpos] 
    embed_list1.pop(maxpos+1)
    score = s[maxpos]
    truth = (gt[maxpos]).item()
    
    # update x
    a = []
    a.append(x[maxpos])
    a.append(x[maxpos+1])
    a.append(truth)
    x[maxpos] = a
    x.pop(maxpos+1)
    return x,embed_list1,score

def traverse_tree(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            a = subtree.label()
            l = len(priority)
            for i in range (0,l):
                if(a.find(priority[i])!= - 1):
                    subtree._label = priority[i]
                    break
                else:
                    if(i==l-1):
                        subtree._label = priority[l-1]
            traverse_tree(subtree)
            
def compute_gtscore(tree, model):
    try:
        tree.label()
    except AttributeError:
        return
    else:
        if(tree.height() <= 2):
            # if its a leaf then return the vector
            a = torch.Tensor(get_embed(tree[0]))
            return torch.Tensor([0]), a, torch.Tensor([0])
        else:
            try:
                sl, pl, ll = compute_gtscore(tree[0], model)
                sr, pr, lr = compute_gtscore(tree[1], model)
                s, p, logprob = model(pl,pr)
                tlist = []
                tlist.append(tree.label())
                gt_val = torch.Tensor(tlist)
                gt_val = gt_val.long()
                logprob = logprob.unsqueeze(dim=0)
                # gt_val = gt_val.unsqueeze(dim=0)
                loss = F.nll_loss(logprob, gt_val)
                s = s + sr + sl
                loss = loss + ll + lr
                return s, p, loss
            except:
                return


def change_labels(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            a = subtree.label()
            subtree._label = final_map[a]
            change_labels(subtree)

def preprocess(y):
    treetransforms.collapse_unary(y,collapsePOS=True)
    treetransforms.chomsky_normal_form(y, horzMarkov=2, vertMarkov=1)
    traverse_tree(y)
    # we got the modified tree so we need to calculate the scores
    change_labels(y)
    y._label = 4

def compute_delta(x,y):
    k = 0.1
    count = 0
    ptree = ParentedTree.convert(x)
    gt_tree = y
    ptree_gt = ParentedTree.convert(gt_tree)

    for subtree in ptree.subtrees():
        len_tree = 0
        for subtree1 in ptree_gt.subtrees():
            len_tree = len_tree+1
            if (subtree == subtree1 ):
                count = count+1

    delta = (len_tree-count)*k
    if(delta < 0):
        delta = 0
    return delta

def train(model,leafmodel, optimizer, total_epochs):
    model.train()
    loss_list = []
    val_loss_list = []
    for t in range(total_epochs):
        tick = time.time()
        total_avg_loss = 0
        print('epoch is .....', t)
        for name in train_data_list:
            print('loaded file ', name)
            filex = treebank.sents(name)
            filey = treebank.parsed_sents(name)
            
            for s in range (0,len(filex)):
                print('working on sentence ', s)
                x = filex[s]
                y = filey[s]
                preprocess(y)
                # embed_x is the list of embedding vectors of x
                embed_x = []
                x_list = []
                l = int(len(x))
                optimizer.zero_grad()

                for i in range (0,l):
                    txlist = []        
                    x[i] = x[i].lower()
                    txlist.append(x[i])
                    tembed = torch.Tensor(get_embed(x[i]))
                    embed_x.append(tembed)

                    pred = leafmodel(embed_x[i])
                    gt = (torch.argmax(pred)).item()
                    txlist.append(gt)                    
                    x_list.append(txlist)

                # we got the (sentence,gt) list, embedding vector list for the leafs 
                xscore = 0.0
                while(len(x_list) != 1):
                    x_list, embed_x, tscore = calculate_score(x_list, embed_x, model)
                    xscore = xscore + tscore
                x_list = str(x_list).replace('[','(').replace(']',')').replace('\'','').replace(',','')
                x_list_tree = Tree.fromstring((x_list))

                # print('xscore is .....', xscore)
                yscore,_,celoss = compute_gtscore(y,model)
                delta_loss = compute_delta(x_list_tree, y)
                flist = []
                flist.append(delta_loss)
                delta_loss = torch.Tensor(flist)
                delta_loss = delta_loss.detach()

                # print('yscore is .....', yscore)
                # print('classification celosss is ....', celoss)
                loss = (xscore + delta_loss - yscore) + celoss 
                loss.backward()
                optimizer.step()
                total_avg_loss = total_avg_loss + loss.item()

        total_avg_loss = total_avg_loss/(len(filex)*len(data_list))
        loss_list.append(total_avg_loss) 
        print('************************************************')
        tock = time.time()
        print('epoch_time ====', (tock-tick))
        val_loss = validate(model, leafmodel)
        val_loss_list.append(val_loss)
        
        torch.save(model, './ckpt/model' + str(t) + '.pt')    
    loss_array = np.array(loss_list)
    val_loss_array = np.array(val_loss_list)
    np.savetxt("train_loss.csv", loss_array, delimiter=",")
    np.savetxt("val_loss.csv", val_loss_array, delimiter=",")
    
def validate(model,leafmodel):
    model.eval()
    loss_list = []
    tick = time.time()
    total_avg_loss = 0
    for name in val_data_list:
        print('loaded file ', name)
        filex = treebank.sents(name)
        filey = treebank.parsed_sents(name)
        
        for s in range (0,len(filex)):
            print('working on sentence ', s)
            x = filex[s]
            y = filey[s]
            preprocess(y)
            # embed_x is the list of embedding vectors of x
            embed_x = []
            x_list = []
            l = int(len(x))

            for i in range (0,l):
                txlist = []        
                x[i] = x[i].lower()
                txlist.append(x[i])
                tembed = torch.Tensor(get_embed(x[i]))
                embed_x.append(tembed)

                pred = leafmodel(embed_x[i])
                gt = (torch.argmax(pred)).item()
                txlist.append(gt)                    
                x_list.append(txlist)

            # we got the (sentence,gt) list, embedding vector list for the leafs 
            xscore = 0.0
            while(len(x_list) != 1):
                x_list, embed_x, tscore = calculate_score(x_list, embed_x, model)
                xscore = xscore + tscore
            
            x_list = str(x_list).replace('[','(').replace(']',')').replace('\'','').replace(',','')
            x_list_tree = Tree.fromstring((x_list))

            # print('xscore is .....', xscore)
            yscore,_,celoss = compute_gtscore(y,model)
            delta_loss = compute_delta(x_list_tree, y)
            flist = []
            flist.append(delta_loss)
            delta_loss = torch.Tensor(flist)
            delta_loss = delta_loss.detach()
            
            # print('xscore is .....', xscore)
            yscore,_,celoss = compute_gtscore(y,model)
            # print('yscore is .....', yscore)
            # print('classification celosss is ....', celoss)
            loss = (xscore - yscore) + celoss + delta_loss
            total_avg_loss = total_avg_loss + loss.item()
    
    total_avg_loss = total_avg_loss/(len(filex)*len(data_list))
    print('validated for this epoch')
    return total_avg_loss

def test():
    model = torch.load('./ckpt/model0.pt')
    leafmodel = LeafNet()
    x = treebank.sents('wsj_0003.mrg')[0]
    y = treebank.parsed_sents('wsj_0003.mrg')[0]
    preprocess(y)
    # embed_x is the list of embedding vectors of x
    embed_x = []
    x_list = []
    l = int(len(x))
    
    for i in range (0,l):
        txlist = []        
        x[i] = x[i].lower()
        txlist.append(x[i])
        tembed = torch.Tensor(get_embed(x[i]))
        embed_x.append(tembed)

        pred = leafmodel(embed_x[i])
        gt = (torch.argmax(pred)).item()
        txlist.append(gt)                    
        x_list.append(txlist)

    # we got the (sentence,gt) list, embedding vector list for the leafs 
    xscore = 0.0
    while(len(x_list) != 1):
        x_list, embed_x, tscore = calculate_score(x_list, embed_x, model)
        xscore = xscore + tscore
    x_list = str(x_list).replace('[','(').replace(']',')').replace('\'','').replace(',','')
    x_list_tree = Tree.fromstring((x_list))

    draw_trees(x_list_tree)
    draw_trees(y)

def main():
    model = NodeNet()
    leafmodel = LeafNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # train(model,leafmodel, optimizer, 7)
    test()
if __name__ == '__main__':
    main()