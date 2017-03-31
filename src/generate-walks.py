'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb', help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--labelled', dest='labelled', action='store_true', help='Boolean specifying (un)labelled. Default is unlabelled.')
    parser.add_argument('--unlabelled', dest='unlabelled', action='store_false')
    parser.set_defaults(labelled=False)

    parser.add_argument('--directed', dest='directed', action='store_true',help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--ignore-popular', dest='ignore_popular', action='store_true', default=False, help = 'Boolean telling whether to ignore popular nodes while doing random walk. Default is not ignoring')
    return parser.parse_args()

def read_graph(ignore_popular):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    elif args.labelled:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('label', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())

        # Edge weight decides the probability of selection in the random walk
        # Lower the weight, lower is the probability of selection
        # If the followers (in degree) is high, we assign lower weight
        if ignore_popular:
            indegrees = dict(G.in_degree_iter())
            for edge in G.edges():
                followers = indegrees[edge[1]]
                if (followers >= 400 and followers < 800):
                    G[edge[0]][edge[1]]['weight'] = 0.7
                elif (followers >= 800 and followers < 2000):
                    G[edge[0]][edge[1]]['weight'] = 0.5
                elif (followers >= 2000):
                    G[edge[0]][edge[1]]['weight'] = 0.2
                else:
                    G[edge[0]][edge[1]]['weight'] = 1
        else:
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks_map = [map(str, walk) for walk in walks]
    with open(args.output, 'w') as fout:
        fout.write(str(walks))
    #model = Word2Vec(walks_map, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    #model.save_word2vec_format(args.output)
    return

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print("Reading graph...")
    nx_G = read_graph(args.ignore_popular)
    print("Reading COMPLETED.\n")
    print("Initializing graph object")
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    print("Initialization COMPLETED")
    print("Calculating transition probabilities")
    G.preprocess_transition_probs()
    print ("Transition probabilities computation COMPLETED")
    print ("Simulating walks...")
    walks = G.simulate_walks(args.num_walks, args.walk_length, args.labelled)
    print ("Walk simulation COMPLETED\n")
    print ("Learning embeddings...\n")
    learn_embeddings(walks)
    print ("Embeddings learning COMPLETED")

if __name__ == "__main__":
    args = parse_args()
    main(args)
