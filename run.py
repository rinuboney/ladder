import tensorflow as tf
import input_data
from ladder import model

layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
#layer_sizes = [784, 500, 10]

L = len(layer_sizes) - 1

num_examples = 60000
num_epochs = 100
num_labelled = 100
batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs
num_l = int(num_examples/(1.0 * num_labelled * batch_size))

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)

y, loss = model(inputs, outputs, layer_sizes)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_step = tf.train.AdamOptimizer(0.002).minimize(loss)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.Session()

init  = tf.initialize_all_variables()
sess.run(init)

print "Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels})

for i in range(num_iter):
    images, labels = mnist.train.next_batch(batch_size)
    if i%num_l == 0:
        sess.run(train_step, feed_dict={inputs: images, outputs: labels[0:1]})
    else:
        sess.run(train_step, feed_dict={inputs: images, outputs: [[]]})
    if i % (num_iter/num_epochs) == 0:
        print "Epoch ", i/(num_examples/batch_size), ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels})
    #if i % (6*num_epochs) == 0:
    #    print i/(6.0*num_epochs), "%"

print "Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels})

sess.close()
