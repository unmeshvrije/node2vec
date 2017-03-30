import pickle, pprint
import sys
from random import shuffle
import argparse
#
#relations
#entities
#train_subs
#valid_subs
#test_subs
#
NUM_BITS_FOR_RELATIONS = 5


def main(datafile, spo, pagerankMap, one_relation, one_domain):
    index_of_subject = 0
    index_of_object = 1
    index_of_predicate = 2
    if spo:
        index_of_object = 2
        index_of_predicate = 1

    pagerank = False
    if len(pagerankMap) != 0:
        pagerank = True

    with open(datafile, 'r') as f:
        lines = f.readlines()
        cntEdges = len(lines)
        if cntEdges < 100000:
            percentageTraining = 80
        else:
            percentageTraining = 99.8
        cntTraining = int((cntEdges * percentageTraining)/100)
        cntTest = int((cntEdges * (float(100 - percentageTraining)/2)) / 100)
        cntValid = cntTest

        testIndex = int(cntTraining + cntTest)

        print ("Total triples : %d\n" % (cntEdges))

        entities_set = set()
        relations_set = set()
        entities_map = {}
        id_to_pagerank_map = {}
        relations_map = {}
        entities = "u'entities': ["
        relations = "u'relations':["
        test = "u'test_subs':["
        trains = "u'train_subs':["
        valid = "u'valid_subs':["

        counter = 0
        identifier = 1
        relationId = 0

        shuffle(lines)

        fwalks = open(datafile + '.combined-relation-entity.edgelist', 'w')
        for index, pair in enumerate(lines):
            counter += 1
            if len(pair.split()) < 2:
                print ("Line [%d] does not represent an edge" % (index))
                continue
            parts = pair.split()
            fromNode = parts[index_of_subject]
            toNode = parts[index_of_object]

            if fromNode not in entities_set:
                entities_set.add(fromNode)
                if pagerank:
                    if fromNode in pagerankMap:
                        entities_map[fromNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                        id_to_pagerank_map[identifier<<NUM_BITS_FOR_RELATIONS+1] = pagerankMap[fromNode]
                    else:
                        entities_map[fromNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                        id_to_pagerank_map[identifier<<NUM_BITS_FOR_RELATIONS+1] = 0.0
                else:
                    entities_map[fromNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                identifier += 1

            if toNode not in entities_set:
                entities_set.add(toNode)
                if pagerank:
                    if toNode in pagerankMap:
                        entities_map[toNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                        id_to_pagerank_map[identifier<<NUM_BITS_FOR_RELATIONS+1] = pagerankMap[toNode]
                    else:
                        entities_map[toNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                        id_to_pagerank_map[identifier<<NUM_BITS_FOR_RELATIONS+1] = 0.0
                else:
                    entities_map[toNode] = (identifier << NUM_BITS_FOR_RELATIONS+1)
                identifier += 1

            if (len(parts) > 2) and not one_relation:
                edgeLabel = parts[index_of_predicate]
                if edgeLabel not in relations_set:
                    relations_set.add(edgeLabel)
                    relations_map[edgeLabel] = relationId
                    relationId +=1
                    if relationId >= 32:
                        print("No more than 31 relations allowed")
                        sys.exit()
                # Here we make two pairs from the triple
                ER = entities_map[fromNode] & relations_map[edgeLabel]
                RE = entities_map[toNode] & relations_map[edgeLabel] | (1 << NUM_BITS_FOR_RELATIONS)

                fwalks.write(str(ER) + " " + str(entities_map[toNode]) + "\n")
                fwalks.write(str(entities_map[fromNode]) + " " + str(RE) + "\n")

        reverse_entities_map = dict(zip(entities_map.values(), entities_map.keys()))
        reverse_relations_map = dict(zip(relations_map.values(), relations_map.keys()))
        with open(datafile + '.entity.map', 'w') as fen:
            fen.write(str(reverse_entities_map))
        with open(datafile + '.relations.map', 'w') as frel:
            frel.write(str(reverse_relations_map))
        with open(datafile + '.pagerank.map', 'w') as fen:
            fen.write(str(id_to_pagerank_map))

        fwalks = open(datafile + '.0-based-entitiy-ids.edgelist', 'w')
        print ("# of identifiers (entities) = %d" % (identifier))
        for index, pair in enumerate(lines):
            parts = pair.split()
            fromNode = parts[index_of_subject]
            toNode = parts[index_of_object]
            edgeLabel = 0
            if len(parts) > 2 and not one_relation:
                edgeLabel = relations_map[parts[index_of_predicate]]

            fwalks.write(str(entities_map[fromNode]) + " " + str(entities_map[toNode]) + " " + str(edgeLabel) + "\n")

            if index < cntTraining-1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            elif index == cntTraining - 1:
                trains += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")],"
            elif index < testIndex-1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            elif index == int(testIndex) - 1:
                test += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")],"
            elif index < cntEdges-1:
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+"),\n"
            else: # index == cntEdges-1
                valid += "(" + str(entities_map[fromNode]) + "," + str(entities_map[toNode]) + ","+str(edgeLabel)+")]"

        for e in entities_set:
            entities += "u'" + str(e) + "'," + "\n"
        entities += "],"

        if len(relations_set) == 0 or one_relation:
            relations += "u'related_to', u'fake'],\n"
        else:
            for r in relations_set:
                relations += "u'" + str(r) + "'," + "\n"
            relations += "],"

    data = "{\n" + entities + relations + trains + test + valid + "}"
    with open(datafile +'.pkl','w') as fout:
        fout.write(data)


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog="SNAP to Python Object format converter", conflict_handler='resolve')
    parser.add_argument('--fin', type=str, help = 'Path to the input file in SNAP (edgelist) format')
    parser.add_argument('--spo', action='store_const', const=True, default=False, help = 'If the input file contains Subject-Predicate-Object format then use this option. Default is Subject-Object-Predicate')
    parser.add_argument('--pagerank', type=str, help = 'Path of the pagerank file mapping')
    parser.add_argument('--one-relation', action='store_const', const=True, default=False, help = 'If you want to exclude relation labels, use this option')
    parser.add_argument('--one-domain', action='store_const', const=True, default=False, help = 'If you want the same ID space (domain) for entities and relations, use this option')
    args = parser.parse_args()
    pagerankMap = {}
    if args.pagerank:
        pagerankMapFile = args.pagerank
        with open(pagerankMapFile, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                entity, pagerank = line.split()
                pagerankMap[entity] = pagerank

    main(args.fin, args.spo, pagerankMap, args.one_relation, args.one_domain)
