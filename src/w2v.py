import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress warnings
import tensorflow as tf
import sys
import math
import random
import collections
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools


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

inputfile = sys.argv[1]  # Some text
inputdata, voc = parseRandomWalks(inputfile)
dim = int(sys.argv[2])  # n dimensions
# n = int(sys.argv[3])  # the number of nodes
n = len(voc) + 1
batch_size = 128
num_sampled = 10    # Number of negative examples to sample.
num_skips = 2         # How many times to reuse an input to generate a label.
skip_window = 3       # How many words to consider left and right.

graph = tf.Graph()
with graph.as_default():
    with tf.device('/cpu:0'):
        # Init embeddings
        embeddings = tf.Variable(
            tf.random_uniform([n, dim], -1.0, 1.0))
        weights = tf.Variable(tf.truncated_normal([n, dim], stddev=1.0 / math.sqrt(dim)))
        biases = tf.Variable(tf.zeros([n]))

    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int64, shape=[batch_size])
    train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Loss function
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights, biases, embed, train_labels, num_sampled, n))
    # We use the SGD optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    # Step 5: Begin training.
    #num_steps = 100001
    num_steps = n-1
    if (n == 1):
        print ("Empty vocabulary.")
        sys.exit()
    num_walks_per_node = int( (len(inputdata) / (n-1)) )

    # Add variable initializer.
    print("Start initialization")
    init = tf.global_variables_initializer()
    average_loss = 0
    current_idx = 0
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")
        start = time.time()
        # Strategies to select batches:
        # 1. Every time, we select from a (random) possibly different walk.
        # 2. We select one batch from each walk (which means there has to be at least "num_steps" walks)
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                inputdata[step*num_walks_per_node],
                0,
                batch_size,
                num_skips,
                skip_window)

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    print("Time epoch", (time.time() - start))
                    start = time.time()
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
