import argparse
import pickle
import sys
import networkx as nx
import numpy as np
from collections import defaultdict as ddict
import operator

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        data = pickle.load(fin)
    return data

########## TAIL PREDICTIONS ##############
# 1
def n_entity_appear_as_tail_of_predicate(G, entity, predicate):
    cnt = 0
    for e in G.in_edges(entity):
        data = G.get_edge_data(e[0], e[1])
        if (data['label'] == predicate):
            cnt += 1
    return cnt

# 2
def n_entity_appear_as_tail_of_entity(G, entity, head):
    cnt = 0
    for e in G.in_edges(entity):
        if e[0] == head:
            cnt += 1
    return cnt

# 3
# if node->entity and node->head, then increase count
def n_entity_related_to_both(G, entity, head):
    cnt = 0
    for node in G.nodes():
        successors = G.successors(node)
        if entity in successors and head in successors:
            cnt += 1
        return cnt

# 4
def n_entity_transitively_similar(G, entity, head):
    cnt = 0;
    head_successors = G.successors(head)
    for node in G.nodes():
        if node not in head_successors:
            continue
        successors = G.successors(node)
        predecessors = G.predecessors(node)
        if entity in successors or entity in predecessors:
            cnt += 1

    return cnt

######### HEAD PREDICTIONS ############
# 1
def n_entity_appear_as_head_of_predicate(G, entity, predicate):
    cnt = 0
    for e in G.out_edges(entity):
        data = G.get_edge_data(e[0], e[1])
        if (data['label'] == predicate):
            cnt += 1
    return cnt

# 2
def n_entity_appear_as_head_of_entity(G, entity, tail):
    cnt = 0
    for e in G.out_edges(entity):
        if e[0] == tail:
            cnt += 1
    return cnt

# 3
# same as tail pedictions
# if node->entity and node->tail, then increase count
def n_entity_related_to_both(G, entity, tail):
    cnt = 0
    for node in G.nodes():
        successors = G.successors(node)
        if entity in successors and tail in successors:
            cnt += 1
        return cnt

# 4
def n_entity_transitively_similar(G, entity, tail):
    cnt = 0;
    tail_predecessors = G.predecessors(tail)
    for node in G.nodes():
        if node not in tail_predecessors:
            continue
        successors = G.successors(node)
        predecessors = G.predecessors(node)
        if entity in successors or entity in predecessors:
            cnt += 1

    return cnt

parser = argparse.ArgumentParser(prog = "Test graph functions")
parser.add_argument('--fdb', type=str, help = "Pickle database")

args = parser.parse_args()
kb = args.fdb

kbRecords = processPickleFile(kb)
N = len(kbRecords['entities'])
M = len(kbRecords['relations'])
train = kbRecords['train_subs'] #+ kbRecords['valid_subs'] + kbRecords['test_subs']
G = nx.DiGraph()

for tuple in train:
    head = tuple[0]
    tail = tuple[1]
    relation = tuple[2]
    G.add_edge(head, tail, label=relation)

print ("# of nodes = %d\n" % len(G.nodes()) )
print ("# of edges = %d\n" % len(G.edges()) )

#for node in G.nodes():
#    print (" node( %d ) = %d\n" % (node, n_entity_appear_as_tail_of_predicate(G, node, 0)) )

arr = [1,2,4,5,7,8,9]
#for node in G.nodes():
#    print (" node ( %d ) = %s\n" % (node, G.in_edges(node, data = True)))
#    print (" node ( %d ) = %s\n" % (node, n_entity_appear_as_tail_of_entity(G, 5, node)))
     #print (" node ( %d ) = %s\n" % (node, n_entity_related_to_both(G, 2, node)))
#     print (" node ( %d ) = %s\n" % (node, n_entity_transitively_similar(G, 20, node)))

indegrees = sorted(G.in_degree_iter(), key=operator.itemgetter(1), reverse=True)
#outdegrees = sorted(G.out_degree_iter(), key=operator.itemgetter(1), reverse=True)
outdegrees = dict(G.out_degree_iter())
for i in indegrees:
    print (i[0], " : ", i[1])

#print(outdegrees[3])
#print (G.predecessors(2))
#print (G.predecessors(20))
