import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress warnings
import tensorflow as tf
import sys
import math
import random
import collections
import pickle
import time
import argparse
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import timeit

def processPickleFile(datafile):
    with open(datafile, 'rb') as fin:
        data = pickle.load(fin)
    return data

# Function to generate a training batch for the skip-gram model.
def generate_batch(data, current_idx, batch_size, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # current_idx should be always within the bounds of one walk.
    # In other words, every batch should contain nodes from the same walk.
    batch = np.ndarray(shape=(batch_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int64)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[current_idx])
        current_idx = (current_idx + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[current_idx])
        current_idx = (current_idx + 1) % len(data)
    return batch, labels


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_random_batch(walks, batch_size, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int64)
    sampleIndex = 0
    for _ in range(batch_size):
        index = random.randint(0, len(walks)-1)
        walkStart = random.randint(0, len(walks[index])-1)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(walks[index][walkStart])
            walkStart = (walkStart + 1) % len(walks[index])
        target = skip_window
        target_to_avoid = [skip_window]
        while target in target_to_avoid:
            target = random.randint(0, span - 1)
        batch[sampleIndex] = buffer[skip_window]
        labels[sampleIndex, 0] = buffer[target]
        sampleIndex += 1

    return batch, labels

def parseText(inputfile):
    words = []
    voc = {}
    idx = 0
    for line in open(inputfile):
        tokens = line.split(' ')
        for w in tokens:
            if w in voc:
                words.append(voc[w])
            else:
                voc[w] = idx
                words.append(idx)
                idx += 1
    return words, voc


#def parseRandomWalks(inputfile):
#    with open(inputfile) as f:
#        data = eval(f.read())
#    chain = itertools.chain(*data)
#    l = [int(el) for el in chain]
#    s = set()
#    for c in l:
#        s.add(c)
#    return l, s

def parseRandomWalks(inputFile):
    with open(inputFile) as f:
        data = eval(f.read())
    chain = itertools.chain(*data)
    l =  [int(el) for el in chain]
    s = set()
    for c in l:
        s.add(c)
    return data, s


parser = argparse.ArgumentParser(prog = "Word2Vec with modified walks")
parser.add_argument('--fin', type=str, help = "Embeddings in the python object text format")
parser.add_argument('--fdb', type=str, help = "Pickle database")
parser.add_argument('--topk', type=int, default=20, help = "TOPK value for evaluation")
parser.add_argument('--dim', type=int, default=50, help = "Number of dimensions")
parser.add_argument('--eval-method', type=str, help = "Evaluation method", default='cosine')
parser.add_argument('--dev', type=str, help = "Whether to run on CPU or GPU", default='gpu')


args = parser.parse_args()
begin = timeit.default_timer()
inputfile = args.fin  # random walks
inputdata, voc = parseRandomWalks(inputfile)
dim = args.dim  # number of dimensions
picklefile = args.fdb # Pickle database containing same IDs as that of the random walks file
dev = args.dev
n = len(voc) #+ 1

data, count, dictionary, reverse_dictionary = build_dataset(voc)

batch_size = 128
num_sampled = 10    # Number of negative examples to sample.
num_skips = 2         # How many times to reuse an input to generate a label.
skip_window = 3       # How many words to consider left and right.
logfile = inputfile + ".log"

valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 1000 # Pick the samples from first 1000 identifiers for entities and relations

# TODO:
# We need two array of identifiers. One for entities and one for relations, so that we can compute cosine distances for both separately
# valid_examples will not be random. They will be taken from the test-dataset of knowledge base.
# 1. Parse test-dataset, make 3 arrays: array of heads, array of predicates and array of tails

testHeads = []
testTails = []
testPreds = []
kbRecords = processPickleFile(picklefile)
test = kbRecords['test_subs']
test_data_size = len(test)
for t in test:
    testHeads.append(t[0])
    testTails.append(t[1])
    testPreds.append(t[2])

# 2. Create corresponding tensor objects to compute cosine distance between
#   A) all heads and all entities
#   B) all relations and all entities
#   C) all tails and all entities
# 3. Compute similarity based on cosine distances and do head/tail predictions
#   A) For tail prediction, find TOP K entities similar to head and TOP K entities similar to relations, and take their intersection
#   B) For head prediction, find TOP K entities similar to tail and TOP K entities similar to relations, and take their intersection
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
batch_heads = np.array(testHeads)
batch_tails = np.array(testTails)
batch_preds = np.array(testPreds)

flog = open(logfile, 'w')
graph = tf.Graph()
with graph.as_default():
    # This is specific to DAS5 node026 and node029, where Titan Pascal GPU is with device id 2
    if dev == 'gpu':
        device_string = '/gpu:2'
    else:
        device_string = '/cpu:0'
    with tf.device(device_string):
        # Init embeddings
        embeddings = tf.Variable(
            tf.random_uniform([n, dim], -1.0, 1.0))
        weights = tf.Variable(tf.truncated_normal([n, dim], stddev=1.0 / math.sqrt(dim)))
        biases = tf.Variable(tf.zeros([n]))

    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int64, shape=[batch_size])
    train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    test_dataset_heads = tf.constant(batch_heads, dtype=tf.int32)
    test_dataset_tails = tf.constant(batch_tails, dtype=tf.int32)
    test_dataset_preds = tf.constant(batch_preds, dtype=tf.int32)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Loss function
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights, biases, embed, train_labels, num_sampled, n))
    # We use the SGD optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

    head_embeddings = tf.nn.embedding_lookup(normalized_embeddings, test_dataset_heads)
    head_similarity = tf.matmul(head_embeddings, normalized_embeddings, transpose_b = True)

    tail_embeddings = tf.nn.embedding_lookup(normalized_embeddings, test_dataset_tails)
    tail_similarity = tf.matmul(tail_embeddings, normalized_embeddings, transpose_b = True)

    pred_embeddings = tf.nn.embedding_lookup(normalized_embeddings, test_dataset_preds)
    pred_similarity = tf.matmul(pred_embeddings, normalized_embeddings, transpose_b = True)

    # Step 5: Begin training.
    #num_steps = 100001
    num_steps = n
    if (n == 0):
        print ("Empty vocabulary.")
        sys.exit()
    num_walks_per_node = int( (len(inputdata) / n) )

    # Add variable initializer.
    print("Start initialization")
    flog.write("Start initialization\n")
    init = tf.global_variables_initializer()
    average_loss = 0
    current_idx = 0
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        flog.write("Initialized\n")
        start = time.time()
        # Strategies to select batches:
        # 1. Every time, we select from a (random) possibly different walk.
        # 2. We select one batch from each walk (which means there has to be at least "num_steps" walks)
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_random_batch(inputdata, batch_size, skip_window)
          #  batch_inputs, batch_labels = generate_batch(
          #      inputdata[step*num_walks_per_node],
          #      0,
          #      batch_size,
          #      num_skips,
          #      skip_window)

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    print("Time epoch", (time.time() - start))
                    flog.write("Time epoch %d\n" % (time.time() - start))
                    start = time.time()
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches
                    print("Average loss at step ", step, ": ", average_loss)
                    flog.write("Average loss at step (%d) : %f\n" %(step, average_loss))
                    average_loss = 0

            if step == num_steps-1:
                # Test loop starts here
                sim = similarity.eval()
                sim_head = head_similarity.eval()
                sim_tail = tail_similarity.eval()
                sim_pred = pred_similarity.eval()
                # for i in xrange(valid_size):
                #     valid_word = reverse_dictionary[valid_examples[i]]
                #     top_k = 8
                #     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #     log_str = "%s : " % valid_word
                #     for k in xrange(top_k):
                #         close_word = reverse_dictionary[nearest[k]]
                #         log_str = "%s %s," % (log_str, close_word)
                #     log_str += "\n"
                #     print (log_str)
                #     flog.write(log_str)
                #
                for i in xrange(test_data_size):
                    # Tail prediction for triple
                    #     head   ,   tail    , predicate >
                    # <test[i][0], test[i][1], test[i][2]>

                    # We don't need the reverse dictionary. It was used to map numbers to possible stringized entities.
                    # Our head, tail and predicates are already numbers.
                    log_str = "Itr# %d\n" % i
                    print (log_str)
                    flog.write(log_str)
                    top_k = 20
                    entity = reverse_dictionary[batch_heads[i]]
                    nearest = (-sim_head[i, :]).argsort()[1:top_k + 1]
                    rank_entity = -1
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        if close_word == batch_tails[i]:
                            rank_entity = k

                    relation = reverse_dictionary[batch_preds[i]]
                    nearest = (-sim_pred[i, :]).argsort()[1:top_k + 1]
                    rank_relation = -1
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        if close_word == batch_tails[i]:
                            rank_relation = k
                    #if relation == 'UNK':
                    #    print (relation, i)
                    log_str = "%s : %d , %s : %d\n" % (entity, rank_entity, relation, rank_relation)
                    print (log_str)
                    flog.write(log_str)

                    # Head prediction for triple
                    # target word is the tail and we will predict the possible head
                    entity = reverse_dictionary[batch_tails[i]]
                    nearest = (-sim_tail[i, :]).argsort()[1:top_k + 1]
                    rank_entity = -1
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        if close_word == batch_heads[i]:
                            rank_entity = k
                    relation = reverse_dictionary[batch_preds[i]]
                    nearest = (-sim_pred[i, :]).argsort()[1:top_k + 1]
                    rank_relation = -1
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        if close_word == batch_heads[i]:
                            rank_relation = k
                    log_str = "%s : %d , %s : %d\n" % (entity, rank_entity, relation, rank_relation)
                    print (log_str)
                    flog.write(log_str)

        final_embeddings = normalized_embeddings.eval()
        end = timeit.default_timer()
        print ("Time to train model = %ds" % (end-begin) )
        flog.write ("Time to train model = %ds\n" % (end-begin) )

        data = ""
        for i, fe in enumerate(final_embeddings):
            data += str(i) + "," + str(fe) + "\n"
        outFile = inputfile + "-embeddings.out"
        with open(outFile, 'w') as fout:
            fout.write(data)
