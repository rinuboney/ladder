import tensorflow as tf
import input_data
import math
import os
import csv
from tqdm import tqdm

layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

L = len(layer_sizes) - 1

num_examples = 60000
num_epochs = 150
num_labeled = 100

lr_decay = 0.67

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs
continue_training = True

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)

bi = lambda inits, size, name: tf.Variable(inits * tf.ones([size]), name=name)
wi = lambda shape, name: tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

shapes = zip(layer_sizes[:-1], layer_sizes[1:])

weights = {'W': [wi(s, "W") for s in shapes],
           'V': [wi(s[::-1], "V") for s in shapes],
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
           'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]}

noise_std = 0.3

denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: tf.concat(0, [l, u])
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.99)
bn_assigns = []

def batch_normalization(batch, mean=None, var=None):
    if mean == None or var == None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

def update_batch_normalization(batch, l):
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l-1].assign(mean)
    assign_var = running_var[l-1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)

def encoder(inputs, noise_std):
    h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std
    d = {}
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
    for l in range(1, L+1):
        print "Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l]
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])
        z_pre_l, z_pre_u = split_lu(z_pre)
        if training:
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            # Training
            if noise_std > 0:
                # Corrupted
                z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean
                z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
	else:
            # Evaluation
    	    mean = ewma.average(running_mean[l-1])
    	    var = ewma.average(running_var[l-1])
            z = join(batch_normalization(z_pre_l, mean, var), batch_normalization(z_pre_u, mean, var))
        if l == L:
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        else:
            h = tf.nn.relu(z + weights["beta"][l-1])
        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print "=== Corrupted Encoder ==="
y_c, corr = encoder(inputs, noise_std)

print "=== Clean Encoder ==="
y, clean = encoder(inputs, 0.0)

print "=== Decoder ==="

def g(z_c, u, size):
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    sigval = wi(0.0, 'c1') + wi(1.0, 'c2') * z_c + wi(0.0, 'c3') * u + wi(0.0, 'c4') * z_c * u
    sigval = tf.sigmoid(sigval)
    z_est = wi(0.0, 'a1') + wi(1.0, 'a2') * z_c + wi(0.0, 'a3') * u + wi(0.0, 'a4') * z_c * u + wi(1.0, 'b1') * sigval
    return z_est

def g_gauss(z_c, u, size):
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est

z_est = {}
d_cost = []
for l in range(L, -1, -1):
    print "Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l]
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        u = tf.matmul(z_est[l+1], weights['V'][l])
    u = batch_normalization(u)
    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    z_est_bn = (z_est[l] - m) / v
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
cost = -tf.reduce_sum(outputs*tf.log(y_N)) / batch_size
loss = cost + u_cost

pred_cost = -tf.reduce_sum(outputs*tf.log(y)) / tf.to_float(tf.shape(y_N)[0])

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.002
learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print "===  Loading Data ==="
mnist = input_data.read_data_sets("MNIST_data", n_labeled = num_labeled, one_hot=True)

saver = tf.train.Saver()

print "===  Starting Session ==="
sess = tf.Session()

i_iter = 0
if continue_training:
    ckpt = tf.train.get_checkpoint_state('checkpoints/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
        i_iter = (epoch_n+1) * (num_examples/batch_size)
        print "Restored Epoch ", epoch_n
    else:
	if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        init  = tf.initialize_all_variables()
        sess.run(init)

train_log = open('train_log', 'ab')
train_log_w = csv.writer(train_log)

print "=== Training ==="
print "Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

for i in tqdm(range(i_iter, num_iter)):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={inputs: images, outputs: labels, training: True})
    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        epoch_n = i/(num_examples/batch_size)
        if (epoch_n+1) >= lr_decay*num_epochs:
            ratio = 1.0 * (num_epochs - (epoch_n+1))
            ratio = max(0, ratio / (num_epochs - lr_decay*num_epochs))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
	log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False})
        train_log_w.writerow(log_i)

print "Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

train_log.close()
sess.close()
