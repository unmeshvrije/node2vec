import tensorflow as tf
# Creates a graph.


def cosTheta(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = tf.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = tf.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)

with tf.device('/gpu:1'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='b')
    prod = tf.reduce_sum(tf.mul(a, b)) # Computes the sum of products of each components of each vector/matrix
    magA = tf.sqrt(tf.reduce_sum(tf.mul(a,a)))
    magB = tf.sqrt(tf.reduce_sum(tf.mul(b,b)))
    r = prod / (magA * magB)

# Creates a session with log_device_placement set to True.
# Setting soft_placement to true will allow tensorflow to choose whichever device is available
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print (sess.run(r))
