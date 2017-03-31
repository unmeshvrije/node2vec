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
DIRECTION_BIT_MASK = (1 << NUM_BITS_FOR_RELATIONS) # Bit 6
COMBINATION_BIT_MASK = (1 << NUM_BITS_FOR_RELATIONS +1) # Bit 7

def main(datafile, spo, one_relation, one_domain):
    index_of_subject = 0
    index_of_object = 1
    index_of_predicate = 2
    if spo:
        index_of_object = 2
        index_of_predicate = 1

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
        edgelist_nodes_set = set()
        entities_map = {}
        edgelist_map = {}
        relations_map = {}
        entities = "u'entities': ["
        relations = "u'relations':["
        test = "u'test_subs':["
        trains = "u'train_subs':["
        valid = "u'valid_subs':["

        counter = 0
        identifier = 1
        relationId = 0
        edgelistId = 0

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
                entities_map[fromNode] = (identifier << NUM_BITS_FOR_RELATIONS+2)
                identifier += 1

            if toNode not in entities_set:
                entities_set.add(toNode)
                entities_map[toNode] = (identifier << NUM_BITS_FOR_RELATIONS+2)
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
                ER = entities_map[fromNode] | relations_map[edgeLabel] | COMBINATION_BIT_MASK
                RE = entities_map[toNode] | relations_map[edgeLabel] | COMBINATION_BIT_MASK | DIRECTION_BIT_MASK

                # Edge list must contains ids from 0, otherwise word2vec does not work on random walks that contain non-0-based ids
                if entities_map[fromNode] not in edgelist_nodes_set:
                    edgelist_nodes_set.add(entities_map[fromNode])
                    edgelist_map[entities_map[fromNode]] = edgelistId
                    edgelistId += 1
                if entities_map[toNode] not in edgelist_nodes_set:
                    edgelist_nodes_set.add(entities_map[toNode])
                    edgelist_map[entities_map[toNode]] = edgelistId
                    edgelistId += 1
                if ER not in edgelist_nodes_set:
                    edgelist_nodes_set.add(ER)
                    edgelist_map[ER] = edgelistId
                    edgelistId += 1
                if RE not in edgelist_nodes_set:
                    edgelist_nodes_set.add(RE)
                    edgelist_map[RE] = edgelistId
                    edgelistId += 1

                fwalks.write(str(edgelist_map[ER]) + " " + str(edgelist_map[entities_map[toNode]]) + "\n")
                fwalks.write(str(edgelist_map[entities_map[fromNode]]) + " " + str(edgelist_map[RE]) + "\n")

        reverse_entities_map = dict(zip(entities_map.values(), entities_map.keys()))
        reverse_relations_map = dict(zip(relations_map.values(), relations_map.keys()))
        reverse_edgelist_map = dict(zip(edgelist_map.values(), edgelist_map.keys()))
        with open(datafile + '.entity.map', 'w') as fen:
            fen.write(str(reverse_entities_map))
        with open(datafile + '.relations.map', 'w') as frel:
            frel.write(str(reverse_relations_map))
        with open(datafile + '.edgelist.map', 'w') as fen:
            fen.write(str(reverse_edgelist_map))

        print ("# of edgelist nodes = %d\n" % (len(reverse_edgelist_map)) )
        fwalks = open(datafile + '.0-based-entitiy-ids.edgelist', 'w')
        print ("# of identifiers (entities) = %d\n" % (identifier))
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
    parser.add_argument('--one-relation', action='store_const', const=True, default=False, help = 'If you want to exclude relation labels, use this option')
    parser.add_argument('--one-domain', action='store_const', const=True, default=False, help = 'If you want the same ID space (domain) for entities and relations, use this option')
    args = parser.parse_args()

    main(args.fin, args.spo, args.one_relation, args.one_domain)
