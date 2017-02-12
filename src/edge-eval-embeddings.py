import pickle, pprint, math
import sys
import pdb
from collections import defaultdict as ddict
import operator
import numpy
import operator
import time
import logging
import itertools

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EVAL-EMBED')
evalMethod = "cosine"

def incoming_neighbours(entity, graph):
    all_lists = list(graph['incoming'][entity].values())
    incoming_neighbours = list(itertools.chain(*all_lists))
    return incoming_neighbours

def outgoing_neighbours(entity, graph):
    all_lists = list(graph['outgoing'][entity].values())
    outgoing_neighbours = list(itertools.chain(*all_lists))
    return outgoing_neighbours

def make_graph(triples, N, M):
    graph_outgoing = [ddict(list) for _ in range(N)]
    graph_incoming = [ddict(list) for _ in range(N)]
    graph_relations_head = [ddict(list)for _ in range(M)]
    graph_relations_tail = [ddict(list)for _ in range(M)]
    for t in triples:
        head = t[0]
        tail = t[1]
        relation = t[2]
        graph_outgoing[head][relation].append(tail)
        graph_incoming[tail][relation].append(head)
        graph_relations_head[relation][head].append(tail)
        graph_relations_tail[relation][tail].append(head)

    return {'outgoing': graph_outgoing, 'incoming': graph_incoming, 'relations_head': graph_relations_head, 'relations_tail':graph_relations_tail}

def find_closest_neighbours(triples, em, TOPK, graph, flog):
    out = []
    computed = []
    cos_dicts = ddict()
    log.info("Number of triples = %d\n" % (len(triples)))
    flog.write("Number of triples = %d\n" % (len(triples)))
    #outgoing = [e for e in range(len(em))]
    for i, t in enumerate(triples):
        head = int(t[0])
        tail = int(t[1])
        relation = int(t[2])
        outgoing = outgoing_neighbours(head, graph)
        flog.write ("Triple(%d %d %d) [%d]: \n" %(head,tail, relation, len(outgoing)))
        log.info ("Triple (%d, %d, %d) [%d] :\n" %(head, tail, relation, len(outgoing)))

        if head in computed:
            cos_dict = cos_dicts[head]
        else:
            cos_dict = ddict()
            for i, o in enumerate(outgoing):
                if evalMethod == "cosine":
                    cos_dict[o] = cosTheta(em[head], em[o])
                else:
                    cos_dict[o] = l1Distance(em[head], em[o])
            cos_dicts[head] = cos_dict
            computed.append(head)

        if evalMethod == "cosine":
            reverse = True
        else:
            reverse = False

        sorted_dict = sorted(cos_dict.items(), key = operator.itemgetter(1), reverse=True)
        #flog.write("Computed cosine distance with neighbours of %d\n" % (head))
        #log.info("Computed cosine distance with neighbours of %d\n" % (head))

        found = False
        for k,v in enumerate(sorted_dict):
            if k == TOPK:
                break
            #flog.write ("%d == %d" % (v[0], tail))
            if v[0] == tail:
                out.append((head, tail, k))
                flog.write("Found (%d, %d, %d)\n" % (head, tail, k))
                log.info("Found (%d, %d, %d)\n" % (head, tail, k))
                found = True
                break
        if k == TOPK:
            out.append((head, tail, -1))
        else:
            # The last element added was the tuple with entity that had <TOPK neighbours
            # Delete this entry and mark it with -2
            if found == True and len(sorted_dict) < TOPK :
                del out[-1]
                out.append((head, tail, -2))
            else:
                if not found:
                    # It should never come here
                    flog.write("!!! %d (outgoing = %d) (sorted_dict = %d) %d\n" % (head, len(outgoing), len(sorted_dict), k))
                    log.info("!!! %d (outgoing = %d) (sorted_dict = %d) %d\n" % (head, len(outgoing), len(sorted_dict), k))


    return out

def processFile(datafile):
    with open(datafile,'r')as fin:
        data = fin.read()

    records = data.split(']')
    # Remove the last element (extra newline)
    del(records[-1])
    embeddings = [[] for _ in range(len(records))]
    for i,r in enumerate(records):
        embeddings_str = r.split(',[')[1].split()
        for e in embeddings_str:
            embeddings[i].append(float(e))

    return numpy.array(embeddings)

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        data = pickle.load(fin)
    return data

def l1Distance(v1, v2):
    distance = 0
    for e1, e2 in zip(v1, v2):
        r = numpy.abs(e1 - e2)
        distance += r
    return distance

# cosine similarity function
# http://stackoverflow.com/questions/1746501/can-someone-give-an-example-of-cosine-similarity-in-a-very-simple-graphical-wa
def cosTheta(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)


if __name__=='__main__':
    if len(sys.argv) < 4:
        print ("Usage: python %s <embeddings.txt> <kb.bin> <TOPK>" % (sys.arg[0]))
        sys.exit()

    logfile = sys.argv[1] + ".log"
    flog = open(logfile, 'w')

    start = time.time()
    embeddings = processFile(sys.argv[1])
    kb = processPickleFile(sys.argv[2])
    flog.write("Time to process files  = %ds\n" % (time.time() - start))
    log.info("Time to process files  = %ds\n" % (time.time() - start))
    TOPK = int(sys.argv[3])

    if (len(sys.argv) > 4):
        evalMethod = sys.argv[4]

    N = len(kb['entities'])
    M = len(kb['relations'])
    training = kb['train_subs']
    valid = kb['valid_subs']
    test = kb['test_subs']
    # this is unfiltered evaluation (because the triples from the training sets are supposed to be guessed correctly)
    dataset = training + valid + test
    start = time.time()
    graph = make_graph(dataset, N, M)
    training_graph = make_graph(training, N, M)
    test_graph = make_graph(test, N, M)
    flog.write("Time to build graphs from triples = %ds\n" %(time.time() - start))
    log.info("Time to build graphs from triples = %ds\n" %(time.time() - start))

    if N != len(embeddings):
        print("Number of entities don't match (embeddings file contains (%d) and pickle database contains (%d))" % (len(embeddings), N))
        sys.exit()

    start = time.time()
    flog.write("Time to make cosine distance matrix = %ds\n" % (time.time() - start))
    log.info("Time to make cosine distance matrix = %ds\n" % (time.time() - start))

    start = time.time()
    #cosines = similarity(embeddings, mat, TOPK, training_graph, test_graph, flog)
    cosines = find_closest_neighbours(test, embeddings, TOPK, graph, flog)
    flog.write("Time to sort and rank best matching objects = %ds\n"%(time.time() - start))
    log.info("Time to sort and rank best matching objects = %ds\n"%(time.time() - start))

    start = time.time()
    outFile = sys.argv[1] + "-" + "TOP-" + str(TOPK) + "-" + evalMethod + ".eval.out"
    data = "{"
    for i, pairs in enumerate(cosines):
        data += str(i) + ": {"
        for p in pairs:
            data += str(p) + " "
        data += "}"
        data += "\n"
    data += "}"
    with open(outFile, 'w') as fout:
        fout.write(data)
    flog.write("Time to write out file = %ds" % (time.time() - start))
    log.info("Time to write out file = %ds" % (time.time() - start))
