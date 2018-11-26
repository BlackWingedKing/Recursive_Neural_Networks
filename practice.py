from nltk.tree import ParentedTree
ptree = ParentedTree.fromstring('(ROOT (S (NP (JJ Congressional) \
    (NNS representatives)) (VP (VBP are) (VP (VBN motivated) \
    (PP (IN by) (NP (NP (ADJ shiny) (NNS money))))))) (. .))')
from nltk import tree, treetransforms
from copy import deepcopy
from nltk.draw.tree import draw_trees

def traverse(t):
    try:
        t.label()
    except AttributeError:
        return
    else:

        if t.height() == 2:   #child nodes
            print (t)
            return

        for child in t:
            traverse(child)

traverse(ptree)
draw_trees(ptree)